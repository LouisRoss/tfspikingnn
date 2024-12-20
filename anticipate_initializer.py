import math
import numpy as np
from base_initializer import BaseInitializer

interconnect_patterns = {
  4: [[-1,0],[0,-1],[1,0],[0,1]],
  8: [[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1]],
  12:[[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-2,0],[0,-2],[2,0],[0,2]]
}

class Initializer(BaseInitializer):
  def __init__(self, configuration):
    super().__init__(configuration)

    self.thickness = configuration.GetThickness()
    self.layer_size = configuration.GetLayerSize()
    self.threshold = configuration.GetThreshold()
    self.interconnectCount = configuration.GetInterconnectCount()

    self.xedgesize, self.yedgesize = self.GenerateSizes()
    self.xedgethick, self.yedgethick = self.GenerateThicknesses()

    # Set default to 16x16 size.
    base1 = 0
    base2 = self.xedgesize

    self.I1 = 0 + base1
    self.I2 = 0 + base2
    self.Inh1 = 1 + base1
    self.Inh2 = 1 + base2
    self.N1 = self.xedgesize - 1 + base1
    self.N2 = self.xedgesize - 1 + base2


  def InitializeInterconnects(self):
    pattern = []
    if self.interconnectCount in interconnect_patterns:
      pattern = interconnect_patterns[self.interconnectCount]

    connections = np.zeros((self.thickness, len(pattern), 1), dtype=np.int32)

    i = 0
    for y in range(self.yedgethick):
      for x in range(self.xedgethick):
        patindex = 0
        pat = np.zeros((len(pattern), 1), dtype=np.int32)
        for connection in pattern:
          xsource = x + connection[0]
          if xsource < 0:
            xsource = 0
          elif xsource >= self.xedgethick:
            xsource = self.xedgethick - 1

          ysource = y + connection[1]
          if ysource < 0:
            ysource = 0
          elif ysource >= self.yedgethick:
            ysource = self.yedgethick - 1

          #print(f'i={i}, y={y}, patindex={patindex}, connection[0]={connection[0]}, connection[1]={connection[1]}')
          source = ysource * self.xedgethick + xsource
          pat[patindex] = [source]

          patindex += 1

        connections[i] = pat

        i += 1

    return connections


  def InitializeConnectionDelays(self):
    delays = super().InitializeConnectionDelays()

    return delays
  

  def InitializeConnections(self):
    layer = super().InitialzieConnections()

    for population in range(self.thickness):
      for instance in range(0, self.yedgesize, 3):
        if instance+1 < self.yedgesize:
          offset = instance * self.xedgesize

          layer[population][offset+self.I1][offset+self.N1] = self.threshold+1
          layer[population][offset+self.I2][offset+self.N2] = self.threshold+1
          layer[population][offset+self.N1][offset+self.N2] = 5
          layer[population][offset+self.N2][offset+self.N1] = 5
          layer[population][offset+self.N1][offset+self.Inh1] = self.threshold+1
          layer[population][offset+self.N2][offset+self.Inh2] = self.threshold+1
          layer[population][offset+self.Inh1][offset+self.I1] = -10
          layer[population][offset+self.Inh2][offset+self.I2] = -10
        
    return layer

  def GenerateSpikes(self, duration):
    initialspikes = super().GenerateSpikes()

    for tick in range(0, duration, 20):
      for population in range(self.thickness):
        for instance in range(0, self.yedgesize, 3):
          if instance+1 < self.yedgesize:
            offset = instance * self.xedgesize

            initialspikes[tick, population, 0, offset+self.I1] = 1
            #initialspikes[tick+1, population, 0, offset+self.I1] = 1
            initialspikes[tick+5, population, 0, offset+self.I2] = 1
            #initialspikes[tick+6, population, 0, offset+self.I2] = 1

    return initialspikes

def MakeInitializer(configuration):
  return Initializer(configuration)
