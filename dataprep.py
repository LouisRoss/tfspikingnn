import os
import json
import csv
import matplotlib.pyplot as plt
import numpy as np
from parselog import ArrayBuilder

class DataPrep:
    baseFolder = '/record/'
    indexFilename = 'index.json'
    connectionsFolder = 'connections/'
    spikesFolder = 'spikes/'
    activationsFolder = 'activations/'
    hebbtimersFolder = 'hebbtimers/'

    connectionSourceFilename = 'fullconnections.dat'
    spikeSourceFilename = 'fullspike.dat'
    activationSourceFilename = 'fullactivations.dat'
    hebblearningSourceFilename = 'fullhebbtimers.dat'

    connectionBaseFilename = 'connection'
    spikeBaseFilename = 'spike'
    activationBaseFilename = 'activation'
    hebbtimerBaseFilename = 'hebbtimer'

    def __init__(self, simulationNumber):
        self.simulationNumber = simulationNumber
        self.simulationExists = os.path.exists(self.MakeSimulationPath())
        self.index = {}

    def __enter__(self):
        if os.path.exists(self.MakeIndexFilePath()):
            os.remove(self.MakeIndexFilePath())
        self.MakeCleanOutputFolder(self.MakeSimulationPath() + DataPrep.connectionsFolder)
        self.MakeCleanOutputFolder(self.MakeSimulationPath() + DataPrep.spikesFolder)
        self.MakeCleanOutputFolder(self.MakeSimulationPath() + DataPrep.activationsFolder)
        self.MakeCleanOutputFolder(self.MakeSimulationPath() + DataPrep.hebbtimersFolder)

        self.index = {
            'simulation': self.simulationNumber,
            'populations': 0,
            'connections': [
            ],
            'spikes': [
            ],
            'activations': [
            ],
            'hebbtimers': [
            ]
        }

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        with open(self.MakeIndexFilePath(), 'w') as indexfile:
            json.dump(self.index, indexfile)


    def MakeSimulationPath(self):
        # The base path for all input and output files in this simulation.
        return DataPrep.baseFolder + 'simulation' + str(self.simulationNumber) + '/'
    
    def MakeIndexFilePath(self):
        # The index file describing all generated output files as well as some simulation parameters.
        return self.MakeSimulationPath() + DataPrep.indexFilename

    # Paths to input files.
    def MakeConnectionsSourceFilePath(self):
        return self.MakeSimulationPath() + DataPrep.connectionSourceFilename
    
    def MakeSpikesSourceFilePath(self):
        return self.MakeSimulationPath() + DataPrep.spikeSourceFilename
    
    def MakeActivationsSourceFilePath(self):
        return self.MakeSimulationPath() + DataPrep.activationSourceFilename

    def MakeHebbtimersSourceFilePath(self):
        return self.MakeSimulationPath() + DataPrep.hebblearningSourceFilename

    # Paths to output files.
    def MakeCleanOutputFolder(self, rootdir):
        if not os.path.exists(rootdir):
            os.makedirs(rootdir)
        else:
            files = os.listdir(rootdir)
            for file in files:
                filepath = os.path.join(rootdir, file)
                if os.path.isfile(filepath):
                    os.remove(filepath)

    def MakeConnectionOutputFilePath(self, population):
        rootdir = self.MakeSimulationPath() + DataPrep.connectionsFolder
        return rootdir + DataPrep.connectionBaseFilename + str(population) + '.csv'

    def MakeSpikeOutputFilePath(self, population):
        rootdir = self.MakeSimulationPath() + DataPrep.spikesFolder
        return rootdir + DataPrep.spikeBaseFilename + str(population) + '.csv'

    def MakeActivationOutputFilePath(self, population):
        rootdir = self.MakeSimulationPath() + DataPrep.activationsFolder
        return rootdir + DataPrep.activationBaseFilename + str(population) + '.csv'

    def MakeHebbtimerOutputFilePath(self, population):
        rootdir = self.MakeSimulationPath() + DataPrep.hebbtimersFolder
        return rootdir + DataPrep.hebbtimerBaseFilename + str(population) + '.csv'

    def BuildCsvFile(self, filename, data):
        print(f'Writing {len(data)} lines into {filename}')
        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(data)


    def BuildConnections(self, debug=False):
        # Convert the raw arrays emitted by TensorFlow to CSV files, one connection per population.
        if not self.simulationExists:
            print(f'No folder for simulation {self.simulationNumber} exists')
            return
        
        if not os.path.exists(self.MakeConnectionsSourceFilePath()):
            print(f'No connection file {self.MakeConnectionsSourceFilePath()} for simulation {self.simulationNumber} exists')
            return
        
        builder = ArrayBuilder(self.MakeConnectionsSourceFilePath())
        builder.Build(debug=debug)
        linedata = builder.linedata

        populationCount = len(linedata)
        if populationCount > self.index['populations']:
            self.index['populations'] = populationCount 

        for pop in range(populationCount):
            connectionFile = self.MakeConnectionOutputFilePath(pop)
            self.index['connections'].append(connectionFile)
            self.BuildCsvFile(connectionFile, linedata[pop])

    def BuildSpikes(self, debug=False):
        # Convert the raw arrays emitted by TensorFlow to CSV files, one set of spikes per population.
        if not self.simulationExists:
            print(f'No folder for simulation {self.simulationNumber} exists')
            return
        
        if not os.path.exists(self.MakeSpikesSourceFilePath()):
            print(f'No spikes file {self.MakeSpikesSourceFilePath()} for simulation {self.simulationNumber} exists')
            return
        
        builder = ArrayBuilder(self.MakeSpikesSourceFilePath())
        builder.Build(debug=debug)
        linedata = builder.linedata

        # An array of populations, each is an array of samples.
        data = []
        if len(linedata) > 0:
            for pop in range(len(linedata[0])):
                data.append([])

        for iteration in range(len(linedata)):
            for pop in range(len(linedata[iteration])):
                data[pop].append(linedata[iteration][pop])

        for pop in range(len(data)):
            spikeFile = self.MakeSpikeOutputFilePath(pop)
            self.index['spikes'].append(spikeFile)
            self.BuildCsvFile(spikeFile, data[pop])


    def BuildActivations(self, debug=False):
        # Convert the raw arrays emitted by TensorFlow to CSV files, one set of activations per population.
        if not self.simulationExists:
            print(f'No folder for simulation {self.simulationNumber} exists')
            return
        
        if not os.path.exists(self.MakeActivationsSourceFilePath()):
            print(f'No activation file {self.MakeActivationsSourceFilePath()} for simulation {self.simulationNumber} exists')
            return
        
        builder = ArrayBuilder(self.MakeActivationsSourceFilePath())
        builder.Build(debug=debug)
        linedata = builder.linedata

        # An array of populations, each is an array of samples.
        data = []
        if len(linedata) > 0:
            for pop in range(len(linedata[0])):
                data.append([])

        for iteration in range(len(linedata)):
            for pop in range(len(linedata[iteration])):
                data[pop].append(linedata[iteration][pop])

        for pop in range(len(data)):
            activationFile = self.MakeActivationOutputFilePath(pop)
            self.index['activations'].append(activationFile)
            self.BuildCsvFile(activationFile, data[pop])


    def BuildHebbianTimers(self, debug=False):
        # Convert the raw arrays emitted by TensorFlow to CSV files, one set of hebbian timers per population.
        if not self.simulationExists:
            print(f'No folder for simulation {self.simulationNumber} exists')
            return
        
        if not os.path.exists(self.MakeHebbtimersSourceFilePath()):
            print(f'No hebbian timing file {self.MakeHebbtimersSourceFilePath()} for simulation {self.simulationNumber} exists')
            return
        
        builder = ArrayBuilder(self.MakeHebbtimersSourceFilePath())
        builder.Build(debug=debug)
        linedata = builder.linedata

        # An array of populations, each is an array of samples.
        data = []
        if len(linedata) > 0:
            for pop in range(len(linedata[0])):
                data.append([])

        for iteration in range(len(linedata)):
            for pop in range(len(linedata[iteration])):
                data[pop].append(linedata[iteration][pop])

        for pop in range(len(data)):
            hebbtimerFile = self.MakeHebbtimerOutputFilePath(pop)
            self.index['hebbtimers'].append(hebbtimerFile)
            self.BuildCsvFile(hebbtimerFile, data[pop])

