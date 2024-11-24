import math
import numpy as np
from base_initializer import BaseInitializer

interconnect_patterns = {
  1: [[-1,0]],
  4: [[-1,0],[0,-1],[1,0],[0,1]],
  8: [[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1]],
  12:[[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-2,0],[0,-2],[2,0],[0,2]]
}

class Initializer(BaseInitializer):
  def __init__(self, configuration):
    super().__init(configuration)

    self.base = 2 * self.xedgesize
    self.Out1 = 0


  def InitializeInterconnects(self):
    """ Generate the interconnects between populations.
        Encode in an array, with one element for each population.
        Each population element in the array is itself an array pointing
        to input populations for the element's population.
    """
    global interconnect_patterns
    return super().InitializerInterconnects(interconnect_patterns)


  def InitializeConnectionDelays(self):
    delays = super().InitializeConnectionDelays()
    delays = delays * 2

    return delays
  

  def InitializeConnections(self):
    """ Set the internal connections between neurons in each of the single populations.
    """
    layer = super().InitialzieConnections()

    for population in range(self.thickness):
      for ycell in range(2, self.yedgesize-2):
        out_cell = ycell*self.xedgesize + self.yedgesize - 1
        for xcell in range(self.xedgesize-2):
            from_cell = ycell*self.xedgesize + xcell
            layer[population][from_cell][from_cell+1] = self.threshold+1
            layer[population][from_cell][out_cell] = 1

    return layer

  def GenerateSpikes(self, duration):
    initialspikes = super().GenerateSpikes()

    for tick in range(0, duration, 40):
      for population in range(self.thickness):
        epoch_tick = tick
        # Spike each input cell in sequence
        for ycell in range(2, self.yedgesize-2):
            spike_cell = ycell*self.xedgesize + 0
            initialspikes[epoch_tick, population, 0, spike_cell] = 1
            epoch_tick += 1

        # At the end of the sequence, spike all the output cells.
        for ycell in range(2, self.yedgesize-2):
            out_cell = ycell*self.xedgesize + self.xedgesize - 1
            initialspikes[epoch_tick, population, 0, out_cell] = 1

    for tick in range(20, duration, 40):
      for population in range(self.thickness):
        epoch_tick = tick
        # Spike each input cell in sequence
        for ycell in range(self.yedgesize-3, 1, -1):
            spike_cell = ycell*self.xedgesize + 0
            initialspikes[epoch_tick, population, 0, spike_cell] = 1
            epoch_tick += 1

    return initialspikes

def MakeInitializer(configuration):
  return Initializer(configuration)
