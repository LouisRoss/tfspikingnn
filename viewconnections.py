import matplotlib.pyplot as plt
import numpy as np
from parseconnections import ConnectionArrayBuilder

base = 16
I1_1=base+0
I1_2=base+1
I1_3=base+2
I1_4=base+3
I1_5=base+4
I1_6=base+5
I1_7=base+6
O1=base+7

base=24
I2_1=base+0
I2_2=base+1
I2_3=base+2
I2_4=base+3
I2_5=base+4
I2_6=base+5
I2_7=base+6
O2=base+7

base=32
I3_1=base+0
I3_2=base+1
I3_3=base+2
I3_4=base+3
I3_5=base+4
I3_6=base+5
I3_7=base+6
O3=base+7

base=40
I4_1=base+0
I4_2=base+1
I4_3=base+2
I4_4=base+3
I4_5=base+4
I4_6=base+5
I4_7=base+6
O4=base+7


connspec = [['I1_1-O1',I1_1,O1], ['I1_2-O1',I1_2,O1], ['I1_3-O1',I1_3,O1], ['I1_4-O1',I1_4,O1], ['I1_5-O1',I1_5,O1], ['I1_6-O1',I1_6,O1], ['I1_7-O1',I1_7,O1], 
            ['I1_1-I1_7',I1_1,I1_7], ['I1_2-I1_7',I1_2,I1_7], ['I1_3-I1_7',I1_3,I1_7], ['I1_4-I1_7',I1_4,I1_7], ['I1_5-I1_7',I1_5,I1_7], ['I1_6-I1_7',I1_6,I1_7], ['O1-I1_7',O1,I1_7]]


simulationnumber = 12
builder = ConnectionArrayBuilder('/media/internal/record/tfspikingnn/simulation' + str(simulationnumber) + '/fullconnections.dat')
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