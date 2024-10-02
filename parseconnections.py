import numpy as np
import csv
import os

class ConnectionArrayBuilder:
    def __init__(self, filename:str) -> None:
        self.filename = filename
        self.csvname = os.path.dirname(self.filename) + '/connections.csv'
        self.depth = 0      # 0 is scalar, 1 is 1-dimension, etc.
        self.debug = False
        self.array = []
        self.linedata = [[], [], [], []]
        self.connspec = []

    def Build(self, connspec, debug=False):
        self.debug = debug

        self.connspec = connspec
        with open(self.csvname, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            header = []
            for col in self.connspec:
                header.append(col[0])

            csvwriter.writerow(header)

        with open(self.filename, 'r') as f:
            linecount = 0
            for line in f:
                #line = f.readline()
                if not line.isspace():
                    #print(line)
                    self.BuildLine(line)
                    linecount += 1
                    if self.depth == 0:
                        #print(f'linedata[0] shape is {np.array(self.linedata[0]).shape}')
                        if len(self.linedata[0]) > 0:
                            print(f'(15,43) = {self.linedata[0][0][15][0][43]}')
                            self.Summarize(self.linedata[0])

        for i in range(len(self.linedata)):
            print(f'linedata[{i}] shape is {np.array(self.linedata[i]).shape}')
        #self.linedata = np.array(self.linedata[0]).squeeze()
        if self.debug:
            print(f'Final array (length {len(self.linedata)}):\n')
            for i in range(len(self.linedata)):
                print(self.linedata[i])

    def Summarize(self, sample):
        with open(self.csvname, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            row = []
            for col in self.connspec:
                row.append(sample[0][col[1]][0][col[2]])

            csvwriter.writerow(row)

    def BuildLine(self, line:str):
            start = self.OpenDepth(line)
            #print(f'Remaining line {line[start:]}')
            self.linedata[self.depth].append(self.ReadLine(line, start))
            currentdepth = self.depth
            self.CloseDepth(line)
            while currentdepth > self.depth:
                if self.debug:
                    print(f'Appending linedata[{currentdepth}] to linedata[{currentdepth-1} and clearing linedata[{currentdepth}]]')
                if currentdepth == 1:
                    self.linedata[currentdepth - 1] = self.linedata[currentdepth]
                else:
                    self.linedata[currentdepth - 1].append(self.linedata[currentdepth])
                self.linedata[currentdepth] = []
                if self.debug:
                    print(self.linedata)
                currentdepth -= 1

    def OpenDepth(self, line:str):
        position = line.find('[')
        while line[position] == '[':
            self.depth += 1
            position += 1

        if self.debug:
            print(f'At open, depth is {self.depth}, data starts at position {position}')
        return position

    def ReadLine(self, line:str, start:int):
        if self.depth == 0:
            return int(line)
        
        values = line[start:].split()
        result = []
        for value in values:
            if value[-1] == ']':
                value = value.split(']')[0]
            result.append(int(value))

        return result


    def CloseDepth(self, line:str):
        position = line.find(']')
        while line[position] == ']':
            self.depth -= 1
            position += 1

        if self.debug:
            print(f'At close, depth is {self.depth}\n')


