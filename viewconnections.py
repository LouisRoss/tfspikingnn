import matplotlib.pyplot as plt
import numpy as np
from parseconnections import ConnectionArrayBuilder

I1 = 10
I2 = 10 + 28
Inh1 = 11
Inh2 = 11 + 28
N1 = 15
N2 = 15 + 28

connspec = [['I1-N1',I1,N1], ['I2-N2',I2,N2],['N1-N2',N1,N2],['N2-N1',N2,N1],['N1-Inh1',N1,Inh1],['N2-Inh2',N2,Inh2],['Inh1-I1',Inh1,I1],['Inh2-I2',Inh2,I2]]


builder = ConnectionArrayBuilder('/record/simulation14/fullconnections.dat')
builder.Build(connspec, debug=False)
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