import re
import os
import tensorflow as tf
import numpy as np
from datetime import datetime
from dataprep import DataPrep


path = '/record/'
basefoldername = 'simulation'
fileparse = r'^([a-zA-Z]+)(\d*)$'

def GetNextSimulationNumber():
  sims = []
  obj = os.scandir(path)
  for entry in obj:
    if entry.is_dir():
      sims.append(int(re.split(fileparse, entry.name)[2]))

  return max(sims) + 1

def MakeSimulationFolder(simulationNumber):
  foldername = path + basefoldername + str(simulationNumber)
  os.makedirs(foldername, exist_ok=True)

  return foldername


class LayerModule(tf.Module):
  def __init__(self, layer_size, thickness, spiketrain, name=None):
    super().__init__(name=name)
    self.is_built = False
    self.layer_size = layer_size
    self.thickness = thickness
    self.spiketrain = spiketrain
    self.tick = tf.Variable(0)
    self.tflayer_size = tf.constant(layer_size, dtype=tf.int32)
    factor = np.ones([self.thickness, 1, self.layer_size], dtype=np.int32)
    factor = factor * 2
    self.factor = tf.constant(factor, dtype=tf.dtypes.int32)
    self.threshold = tf.constant(12)
    self.hebblearning = tf.constant(2)
    hebbdelay = np.ones([self.thickness, 1, self.layer_size], dtype=np.int32)
    hebbdelay = hebbdelay * 7
    self.hebbdelay = tf.constant(hebbdelay, dtype=tf.dtypes.int32)
    layer = self.InitializeConnections()
    self.connections = tf.Variable(layer, name='connections', trainable=False)
    self.delaytimes = tf.Variable(tf.zeros([self.thickness, 1, self.layer_size], dtype=tf.dtypes.int32), dtype=tf.dtypes.int32, name='delaytimes', trainable=False)
    self.delayguards = tf.Variable(tf.zeros([self.thickness, 1, self.layer_size], dtype=tf.dtypes.int32), dtype=tf.dtypes.int32, name='delayguards', trainable=False)

  def HebbLearning(self):
    # Broadcast hebbtimers across all rows of a workspace, then filter out columns that are not spiking this tick.
    activehebb = tf.zeros([self.thickness, self.layer_size,self.layer_size], dtype=tf.dtypes.int32) + tf.maximum(tf.transpose(self.hebbtimers, perm=[0,2,1]), self.hebblearning) * self.spikes

    # Add resulting columns container hebbtimers to existing connections, capping any connection at the spike threshold.
    self.connections.assign(tf.cast(tf.minimum(self.connections + activehebb, self.threshold), tf.int32))

    self.hebbtimers.assign_add((self.spikes * self.hebbdelay))
    self.hebbtimers.assign(tf.cast(tf.maximum(tf.subtract(self.hebbtimers, 1), 0), tf.int32))

  def __call__(self, datafolder, log=False):
    # Create variables on first call.
    if not self.is_built:
      self.potentials = tf.Variable(tf.zeros([self.thickness, 1, self.layer_size], dtype=tf.dtypes.int32), dtype=tf.dtypes.int32, name='potentials', trainable=False)
      self.decayedpotentials = tf.Variable(tf.zeros([self.thickness, 1, self.layer_size], dtype=tf.dtypes.int32), dtype=tf.dtypes.int32, name='decayedpotentials', trainable=False)
      self.resets = tf.Variable(tf.zeros([self.thickness, 1, self.layer_size], dtype=tf.dtypes.int32), dtype=tf.dtypes.int32, name='resets', trainable=False)
      self.hebbtimers = tf.Variable(tf.zeros([self.thickness, 1, self.layer_size], dtype=tf.dtypes.int32), dtype=tf.dtypes.int32, name='hebbtimers', trainable=False)
      initialspikes = np.zeros((self.thickness, 1, self.layer_size), dtype=np.int32)
      self.spikes = tf.Variable(tf.cast(initialspikes, tf.dtypes.int32), trainable=False)

      self.is_built = True


    # potential(j) += SUM(ij)[spike(i) @ connection(ij)]
    self.potentials.assign((self.spikes @ self.connections) + self.decayedpotentials)
    #self.potentials.assign(((self.spiketrain[self.tick] + self.spikes) @ self.connections) + self.decayedpotentials)
    #self.tick.assign_add(1)

    # Spike if above threshold, but only if delaytime is exhausted.
    #self.spikes.assign(tf.cast(tf.greater_equal(self.potentials, self.threshold), tf.int32))
    self.spikes.assign(tf.cast(tf.greater_equal(tf.multiply(self.potentials, self.delayguards), self.threshold), tf.int32))
    self.HebbLearning()
    self.spikes.assign(tf.minimum(self.spikes + self.spiketrain[self.tick], 1))
    self.tick.assign_add(1)

    # delaytime(i) = delaytime(i) + 8 if spike(i) else delaytime(i)
    self.delaytimes.assign_add(tf.multiply(self.spikes, 8))

    # delayguards can be used to filter out cells with delaytime > 0.  Delayguard is 1 if delaytime <= 0.
    self.delayguards.assign(tf.cast(tf.less_equal(self.delaytimes, 0), tf.int32))

    # decaypotential(i) = potential(i) / 2, or 0 if delaytime is delaying.
    self.decayedpotentials.assign(tf.cast(tf.divide(tf.multiply(self.potentials, self.delayguards), self.factor), dtype=tf.dtypes.int32))

    self.delaytimes.assign(tf.cast(tf.maximum(tf.subtract(self.delaytimes, 1), 0), tf.int32))

    if log:
      tf.print(self.spikes, summarize=-1, sep=',', output_stream= 'file://' + datafolder + 'fullspike.dat')
      tf.print(self.potentials, summarize=-1, sep=',', output_stream= 'file://' + datafolder + 'fullactivations.dat')
      tf.print(self.hebbtimers, summarize=-1, sep=',', output_stream= 'file://' + datafolder + 'fullhebbtimers.dat')

    return self.spikes
  

  def InitializeConnections(self):
      #layer = tf.cast(tf.random.normal([self.thickness, self.layer_size, self.layer_size], mean=0.0, stddev=10.0), tf.dtypes.int32)
      layer = np.random.randint(-25, high=25, size=(self.thickness, self.layer_size, self.layer_size))
      """
      layer = np.zeros((self.thickness, self.layer_size, self.layer_size))
      """
      for i in range(self.layer_size):
        for j in range(self.layer_size):
          layer[0][i][j] = 0
          layer[1][i][j] = 0

      I1 = 10
      I2 = 10 + 28
      Inh1 = 11
      Inh2 = 11 + 28
      N1 = 15
      N2 = 15 + 28

      for i in range(0, 20*28, 4*28):
        layer[0][i+I1][i+N1] = self.threshold+1
        layer[0][i+I2][i+N2] = self.threshold+1
        layer[0][i+N1][i+N2] = 5
        layer[0][i+N2][i+N1] = 5
        layer[0][i+N1][i+Inh1] = self.threshold+1
        layer[0][i+N2][i+Inh2] = self.threshold+1
        layer[0][i+Inh1][I1] = -10
        layer[0][i+Inh2][I2] = -10

      layer = tf.cast(layer, tf.dtypes.int32)
      return layer

class PopulationModule(tf.Module):
  def __init__(self, layer_size, iterations, thickness=1, name=None):
    super().__init__(name=name)
    self.duration = 1000
    self.layer_size = layer_size
    self.iterations = tf.constant(iterations)
    self.thickness = tf.constant(thickness)
    thicks = []
    for pop in range(layer_size):
      thicks.append(f'file://data/population{pop}.csv')
    self.thickname = tf.Variable(thicks, dtype=tf.dtypes.string, name='populationnames', trainable=False)

    spiketrain = self.GenerateSpikes()
    self.population = LayerModule(layer_size=self.layer_size, thickness=self.thickness, spiketrain=spiketrain, name="population")
    self.spikevalues = tf.Variable(tf.zeros([self.thickness, 1, self.layer_size], dtype=tf.dtypes.int32), trainable=False)

  def GenerateSpikes(self):
    initialspikes = np.zeros((self.duration, self.thickness, 1, self.layer_size), dtype=np.int32)

    I1 = 10
    I2 = 10 + 28

    for tick in range(0, self.duration, 20):
      for i in range(0, 20*28, 4*28):
        initialspikes[tick, 0, 0, i+I1] = 1
        #initialspikes[tick+1, 0, 0, i+I1] = 1
        initialspikes[tick+5, 0, 0, i+I2] = 1
        #initialspikes[tick+6, 0, 0, i+I2] = 1

    return tf.Variable(tf.cast(initialspikes, tf.dtypes.int32), trainable=False)

  @tf.function
  def __call__(self, datafolder, log=False):
    if log:
        tf.print(self.population.connections, summarize=-1, sep=',', output_stream= 'file://' + datafolder + 'fullconnections.dat')

    i = tf.constant(0)
    while i < self.iterations:
      self.spikevalues.assign(self.population(datafolder, log))
      i = i+1



device_name = tf.test.gpu_device_name()
if not device_name:
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

def TracePopulationModel():
  # Set up logging.
  stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
  logdir = "logs/neuronfunc/%s" % stamp
  writer = tf.summary.create_file_writer(logdir)

  # Create a new model to get a fresh trace
  # Otherwise the summary will not see the graph.
  population_model = PopulationModule(name="populations")

  # Bracket the function call with
  # tf.summary.trace_on() and tf.summary.trace_export().
  tf.summary.trace_on(graph=True)
  tf.profiler.experimental.start(logdir)
  # Call only one tf.function when tracing.
  z = print(population_model())
  with writer.as_default():
    tf.summary.trace_export(
        name="my_func_trace",
        step=0,
        profiler_outdir=logdir)

def Run():
  #tf.debugging.set_log_device_placement(True)

  simulationNumber = GetNextSimulationNumber()
  datafolder = MakeSimulationFolder(simulationNumber) + '/'

  iterationCount = 1000
  population_model = PopulationModule(layer_size=784, iterations=iterationCount, thickness=32, name="populations")
  #c = population_model()
  #print(c)
  #tf.debugging.set_log_device_placement(False)

  startTime = datetime.now()
  population_model(datafolder=datafolder, log=True)
  endTime = datetime.now()
  duration = endTime - startTime
  print(f'Run time for {iterationCount} iterations: {duration} or {duration.seconds / iterationCount} s/iter')

  with DataPrep(simulationNumber) as prep:
    prep.BuildConnections()
    prep.BuildSpikes()
    prep.BuildActivations()
    prep.BuildHebbianTimers()

Run()
