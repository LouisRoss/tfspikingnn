import matplotlib.pyplot as plt
import numpy as np
from parselog import ArrayBuilder

builder = ArrayBuilder('data/fullspike1.csv')
builder.Build(debug=False)
linedata = builder.linedata

#print(f'Shape: {linedata.shape}, size of dimension 0: {len(linedata)}, dimension 1: {len(linedata[0])}, dimension 2: {len(linedata[0][0])}, dimension 3: {len(linedata[0][0][0])}')
data1 = []
data2 = []
data3 = []
data4 = []
data5 = []
data6 = []
data7 = []
data8 = []
data = []
population = 0
print(f'Shape: {linedata.shape}')
for iteration in linedata:
    data.append([])

for iteration in range(len(linedata)):
    #print(f'{iteration[0][0][0]}\n')
    #print(f'{iteration.squeeze()[0]}\n')
    print(f'{linedata[iteration][population]}\n')

    for pop in range(8):
        data[pop].append(linedata[iteration][population+pop])
    """
    data1.append(iteration[population])
    data2.append(iteration[population+1])
    data3.append(iteration[population+2])
    data4.append(iteration[population+3])
    data5.append(iteration[population+4])
    data6.append(iteration[population+5])
    data7.append(iteration[population+6])
    data8.append(iteration[population+7])
    """

fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(10,50), layout="constrained")
print(axs)

pop = 0
for ax in axs.flat:
    ax.imshow(data[pop])
    pop += 1
"""
im = ax1.imshow(data1)
im = ax2.imshow(data2)
im = ax3.imshow(data3)
im = ax4.imshow(data4)
im = ax5.imshow(data5)
im = ax6.imshow(data6)
im = ax7.imshow(data7)
im = ax8.imshow(data8)
ax1.set_title('Spikes vs time')
"""
plt.show()
