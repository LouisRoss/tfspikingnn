import matplotlib.pyplot as plt
import numpy as np
from parselog import ArrayBuilder

simulationnumber = 8
#builder = ArrayBuilder('/media/internal/record/tfspikingnn/simulation' + str(simulationnumber) + '/fullspike.dat')
builder = ArrayBuilder('/media/internal/record/tfspikingnn/simulation' + str(simulationnumber) + '/fullactivations.dat')
#builder = ArrayBuilder('/media/internal/record/tfspikingnn/simulation' + str(simulationnumber) + '/fullhebbtimers.dat')
builder.Build(debug=False)
linedata = builder.linedata

data = []
population = 0
print(f'Shape: {linedata.shape}')
duration = min(len(linedata), 10)
print(f'Using the first {duration} ticks')
for iteration in range(duration):
    data.append([])

for iteration in range(duration):
    #print(f'{iteration[0][0][0]}\n')
    #print(f'{linedata[iteration][population]}\n')

    for pop in range(8):
        data[pop].append(linedata[iteration][population+pop])

"""
fig, ax = plt.subplots(figsize=(100,100))
ax.imshow(data[0])
"""
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(10,50), layout="constrained")
print(axs)

pop = 0
for ax in axs.flat:
    ax.imshow(data[pop])
    pop += 1

plt.show()