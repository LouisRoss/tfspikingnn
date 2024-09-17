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

    connectionSourceFilename = 'fullconnections.dat'
    spikeSourceFilename = 'fullspike.dat'
    activationSourceFilename = 'fullactivations.dat'

    connectionBaseFilename = 'connection'
    spikeBaseFilename = 'spike'
    activationBaseFilename = 'activation'

    def __init__(self, simulationNumber):
        self.simulationNumber = simulationNumber
        self.simulationExists = os.path.exists(self.MakeSimulationPath())
        self.index = {}

    def __enter__(self):
        indexLoaded = False
        if os.path.exists(self.MakeIndexFilePath()):
            try:
                self.index = json.loads(self.MakeIndexFilePath())
                indexLoaded = True
            except:
                pass

        if not indexLoaded:
            self.index = {
                'simulation': self.simulationNumber,
                'populations': 0,
                'connections': [
                ],
                'spikes': [
                ],
                'activations': [
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

    # Paths to output files.    
    def MakeConnectionOutputFilePath(self, population):
        if not os.path.exists(self.MakeSimulationPath() + DataPrep.connectionsFolder):
            os.makedirs(self.MakeSimulationPath() + DataPrep.connectionsFolder)
        return self.MakeSimulationPath() + DataPrep.connectionsFolder + DataPrep.connectionBaseFilename + str(population) + '.csv'

    def MakeSpikeOutputFilePath(self, population):
        if not os.path.exists(self.MakeSimulationPath() + DataPrep.spikesFolder):
            os.makedirs(self.MakeSimulationPath() + DataPrep.spikesFolder)
        return self.MakeSimulationPath() + DataPrep.spikesFolder + DataPrep.spikeBaseFilename + str(population) + '.csv'

    def MakeActivationOutputFilePath(self, population):
        if not os.path.exists(self.MakeSimulationPath() + DataPrep.activationsFolder):
            os.makedirs(self.MakeSimulationPath() + DataPrep.activationsFolder)
        return self.MakeSimulationPath() + DataPrep.activationsFolder + DataPrep.activationBaseFilename + str(population) + '.csv'


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
            with open(connectionFile, 'w') as csvfile:
                csvwriter = csv.writer(csvfile)
                for line in linedata[pop]:
                    csvwriter.writerow(line)

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
            with open(spikeFile, 'w') as csvfile:
                csvwriter = csv.writer(csvfile)
                for line in data[pop]:
                    csvwriter.writerow(line)


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
            with open(activationFile, 'w') as csvfile:
                csvwriter = csv.writer(csvfile)
                for line in data[pop]:
                    csvwriter.writerow(line)

