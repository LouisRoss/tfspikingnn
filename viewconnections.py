import matplotlib.pyplot as plt
import numpy as np
from parselog import ArrayBuilder

builder = ArrayBuilder('/media/internal/record/tfspikingnn/simulation10/fullconnections.dat')
builder.Build(debug=False)
linedata = builder.linedata
"""
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(10,50), layout="constrained")
print(axs)

pop = 0
for ax in axs.flat:
    ax.imshow(linedata[pop])
    pop += 1
#plt.show()
"""
fig, ax = plt.subplots(figsize=(100,100))
ax.imshow(linedata[0])
#ax.imshow(linedata[0])
#plt.savefig('images/connections.png', dpi=72)
plt.show()