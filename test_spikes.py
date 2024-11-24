import unittest
import numpy as np
import tensorflow as tf
from neuronconfiguration import NeuronConfiguration
from base_initializer import BaseInitializer

# System under test
from neuronlayer import LayerModule


basic_configuration = {
    "name": "Basic Test ",
    "iterationCount": 10,
    "layerSize": 16,
    "thickness": 4,
    "threshold": 12,
    "interconnectCount": 4,
    "outputWidth": 2,
    "selectedInitializer": 0,
    "initializers": [
        "sequence_initializer"
    ]
}


interconnect_patterns = {
  1: [[-1,0]],
  4: [[-1,0],[0,-1],[1,0],[0,1]],
  8: [[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1]],
  12:[[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-2,0],[0,-2],[2,0],[0,2]]
}

class Initializer(BaseInitializer):
  def __init__(self, configuration):
    super().__init__(configuration)

    self.base = 2 * self.xedgesize
    self.Out1 = 0


  def InitializeInterconnects(self):
    """ Generate the interconnects between populations.
        Encode in an array, with one element for each population.
        Each population element in the array is itself an array pointing
        to input populations for the element's population.
    """
    global interconnect_patterns
    return super().InitializeInterconnects(interconnect_patterns)


  def InitializeConnectionDelays(self):
    delays = super().InitializeConnectionDelays()

    return delays
  

  def InitializeConnections(self):
    """ Set the internal connections between neurons in each of the single populations.
    """
    layer = super().InitializeConnections()

    for population in range(self.thickness):
      for ycell in range(1, self.yedgesize-1):
        from_cell = ycell*self.xedgesize
        to_cell = ycell*self.xedgesize + self.yedgesize - 1
        layer[population][from_cell][to_cell] = 5
        layer[population][from_cell+1][to_cell] = 5
        layer[population][from_cell+2][to_cell] = self.threshold -3

    return layer

  def GenerateSpikes(self, duration):
    initialspikes = super().GenerateSpikes(duration)

    for tick in range(0, duration, 40):
      for population in range(self.thickness):
        epoch_tick = tick
        # Spike each input cell in sequence
        for ycell in range(1, self.yedgesize-1):
            spike_cell = ycell*self.xedgesize + 0
            initialspikes[epoch_tick, population, 0, spike_cell] = 1

            spike_cell = ycell*self.xedgesize
            to_cell = ycell*self.xedgesize + self.yedgesize - 1
            initialspikes[epoch_tick, population, 0, spike_cell] = 1
            initialspikes[epoch_tick+1, population, 0, spike_cell+1] = 1
            initialspikes[epoch_tick+2, population, 0, spike_cell+2] = 1
            initialspikes[epoch_tick+3, population, 0, to_cell] = 1
            epoch_tick += 1


    return initialspikes
  
  def CheatSheet(self):
    expected_spikes = [
       [[0, 0, 0, 0,  1, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0]],
       [[0, 0, 0, 0,  0, 1, 0, 0,  1, 0, 0, 0,  0, 0, 0, 0]],
       [[0, 0, 0, 0,  0, 0, 1, 0,  0, 1, 0, 0,  0, 0, 0, 0]],
       [[0, 0, 0, 0,  0, 0, 0, 1,  0, 0, 1, 0,  0, 0, 0, 0]],
       [[0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 1,  0, 0, 0, 0]],
       [[0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0]]
    ]

    expected_potentials = [
       [[0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0]],
       [[0, 0, 0, 0,  0, 0, 0, 5,  0, 0, 0, 0,  0, 0, 0, 0]],
       [[0, 0, 0, 0,  0, 0, 0, 7,  0, 0, 0, 5,  0, 0, 0, 0]],
       [[0, 0, 0, 0,  0, 0, 0, 12, 0, 0, 0, 7,  0, 0, 0, 0]],
       [[0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 12, 0, 0, 0, 0]],
       [[0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0]]
    ]

    return (expected_spikes, expected_potentials)
     

def TensorEqual(expected, actual):
   equals = tf.cast(tf.equal(expected, actual), tf.int32)
   tensorsequal = (tf.reduce_max(1 - equals) == 0)
   if not tensorsequal:
      print(f'expected: {expected}')
      print(f'actual:   {actual}')

   return tensorsequal

class TestSpikingModel(unittest.TestCase):

    def setUp(self):
        self.configuration = NeuronConfiguration('', basic_configuration)
        if not self.configuration.valid:
            print(f'Configuration is not valid')
            self.layer = None

        if self.configuration.valid:
            self.initializer = Initializer(self.configuration)
            self.layer = LayerModule(self.configuration, self.initializer)

    def test_configured(self):
      self.assertTrue(self.configuration.valid)

    def test_correctinitializer(self):
      self.assertEqual(0, self.layer.configuration.GetSelectedInitializer())

    def test_correctdimensions(self):
      self.assertEqual(self.configuration.configuration['iterationCount'],    self.layer.iterations)
      self.assertEqual(self.configuration.configuration['layerSize'],         self.layer.layer_size)
      self.assertEqual(self.configuration.configuration['thickness'],         self.layer.thickness)
      self.assertEqual(self.configuration.configuration['outputWidth'],       self.layer.outputwidth)
      self.assertEqual(self.configuration.configuration['interconnectCount'], self.layer.interconnectCount)
      self.assertEqual(self.layer.outputwidth * self.layer.interconnectCount, self.layer.inputwidth)
      #print(self.layer.connections[0])

    def test_singlestep(self):
      # Execute a single tick.
      self.layer('')
      self.assertTrue(TensorEqual(self.initializer.InitializeConnectionDelays(), self.layer.connection_delays))
      self.assertTrue(TensorEqual(self.initializer.InitializeConnections(), self.layer.connections))

    def test_spikes_and_potentials(self):
      (expected_spikes, expected_potentials) = self.initializer.CheatSheet()
      self.assertEqual(len(expected_spikes), len(expected_potentials))
 
      print(self.layer.connection_delays[0])
      for i in range(len(expected_spikes)):
        expected_spike = expected_spikes[i]
        expected_potential = expected_potentials[i]

        # Execute a tick for every expected element in the cheatsheet.
        self.layer('')

        # All populations connected identically and spike identically.
        for population in range(self.initializer.thickness):
          self.assertTrue(TensorEqual(expected_spike, self.layer.spikes[population]))
          self.assertTrue(TensorEqual(expected_potential, self.layer.potentials[population]))
          #print(self.layer.spikes[population,0])
          #print(self.layer.potentials[population])

        print(self.layer.spikes[0,0])
        print(self.layer.connection_delays[0,4])

      print(self.layer.connection_delays[0])

    def test_delay_adjustment(self):
      (expected_spikes, _) = self.initializer.CheatSheet()

      original_delay_times = self.layer.connection_delays

      for i in range(len(expected_spikes)):
        # Execute a tick for every expected element in the cheatsheet.
        self.layer('')
