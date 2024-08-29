import matplotlib.pyplot as plt
import numpy as np
from parselog import ArrayBuilder

builder = ArrayBuilder('data/fullconnections.csv')
builder.Build(debug=False)
linedata = builder.linedata

fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(10,50), layout="constrained")
print(axs)

pop = 0
for ax in axs.flat:
    ax.imshow(linedata[pop])
    pop += 1

plt.show()
