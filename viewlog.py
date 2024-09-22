import matplotlib.pyplot as plt
import numpy as np
from parselog import ArrayBuilder

builder = ArrayBuilder('/media/internal/record/tfspikingnn/simulation0/fullspike.dat')
builder.Build(debug=False)
linedata = builder.linedata

data = []
population = 0
print(f'Shape: {linedata.shape}')
for iteration in range(len(linedata)):
    data.append([])

for iteration in range(len(linedata)):
    #print(f'{iteration[0][0][0]}\n')
    print(f'{linedata[iteration][population]}\n')

    for pop in range(8):
        data[pop].append(linedata[iteration][population+pop])

fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(10,50), layout="constrained")
print(axs)

pop = 0
for ax in axs.flat:
    ax.imshow(data[pop])
    pop += 1

plt.show()
