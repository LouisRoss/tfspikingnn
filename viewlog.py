import numpy as np
from parselog import ArrayBuilder

builder = ArrayBuilder('data/fullspike1.csv')
builder.Build(debug=False)
linedata = builder.linedata

#print(f'Shape: {linedata.shape}, size of dimension 0: {len(linedata)}, dimension 1: {len(linedata[0])}, dimension 2: {len(linedata[0][0])}, dimension 3: {len(linedata[0][0][0])}')
print(f'Shape: {linedata.shape}')
for iteration in linedata:
    #print(f'{iteration[0][0][0]}\n')
    #print(f'{iteration.squeeze()[0]}\n')
    print(f'{iteration[0]}\n')
