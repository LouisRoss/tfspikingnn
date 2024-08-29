import tensorflow as tf
import numpy as np
from datetime import datetime
from datetime import datetime


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
    layer = self.InitializeConnections()
    self.connections = tf.Variable(layer, name='connections', trainable=False)
    self.delaytimes = tf.Variable(tf.zeros([self.thickness, 1, self.layer_size], dtype=tf.dtypes.int32), dtype=tf.dtypes.int32, name='delaytimes', trainable=False)
    self.delayguards = tf.Variable(tf.zeros([self.thickness, 1, self.layer_size], dtype=tf.dtypes.int32), dtype=tf.dtypes.int32, name='delayguards', trainable=False)

  def __call__(self):
    # Create variables on first call.
    if not self.is_built:
      self.potentials = tf.Variable(tf.zeros([self.thickness, 1, self.layer_size], dtype=tf.dtypes.int32), dtype=tf.dtypes.int32, name='potentials', trainable=False)
      self.decayedpotentials = tf.Variable(tf.zeros([self.thickness, 1, self.layer_size], dtype=tf.dtypes.int32), dtype=tf.dtypes.int32, name='decayedpotentials', trainable=False)
      self.resets = tf.Variable(tf.zeros([self.thickness, 1, self.layer_size], dtype=tf.dtypes.int32), dtype=tf.dtypes.int32, name='resets', trainable=False)
      initialspikes = np.zeros((self.thickness, 1, self.layer_size), dtype=np.int32)
      """
      initialspikes[0, 0, 317] = 1
      initialspikes[0, 0, 320] = 1
      initialspikes[1, 0, 4] = 1
      initialspikes[2, 0, 1] = 1
      initialspikes[3, 0, 1] = 1
      initialspikes[4, 0, 1] = 1
      initialspikes[5, 0, 1] = 1
      initialspikes[6, 0, 1] = 1
      initialspikes[7, 0, 1] = 1
      """
      self.spikes = tf.Variable(tf.cast(initialspikes, tf.dtypes.int32), trainable=False)

      self.is_built = True


    # potential(j) += SUM(ij)[spike(i) @ connection(ij)]
    self.potentials.assign((self.spikes @ self.connections) + self.decayedpotentials)

    # spike(i) = 1 if potential(i) > 12 else 0
    self.spikes.assign(tf.cast(tf.greater(self.potentials, 12), tf.int32))
    self.spikes.assign_add(self.spiketrain[self.tick])
    self.tick.assign_add(1)

    # delaytime(i) = delaytime(i) + 8 if spike(i) else delaytime(i)
    self.delaytimes.assign_add(tf.multiply(self.spikes, 8))

    # decaypotential(i) = potential(i) / 2
    self.decayedpotentials.assign(tf.cast(tf.divide(self.potentials, self.factor), dtype=tf.dtypes.int32))

    # decaypotential(i) = 0 if delaytime(i) > 0 else decaypotential(i)
    self.delayguards.assign(tf.cast(tf.less_equal(self.delaytimes, 0), tf.int32))
    self.decayedpotentials.assign(tf.multiply(self.decayedpotentials, self.delayguards))

    # delaytime(i) = delaytime(i) - 1 if delaytime(i) > 0 else delaytime(i)
    self.delayguards.assign(tf.cast(tf.subtract(1, self.delayguards), tf.int32))
    self.delaytimes.assign(tf.cast(tf.subtract(self.delaytimes, self.delayguards), tf.int32))

    #return self.delayguards
    #return self.delaytimes
    #return self.potentials
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

      for i in range(10, self.layer_size - 10):
        layer[0][i][i+1] = 11
        layer[0][i][i+4] = 8
        layer[0][i][i+6] = 4
        layer[1][i][self.layer_size - (i+2)] = 13
        layer[1][i][self.layer_size - (i+3)] = 5
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
    for tick in range(0, 20, 2):
      initialspikes[tick, 0, 0, 317] = 1
      initialspikes[tick+1, 0, 0, 320] = 1

    initialspikes[0, 1, 0, 4] = 1
    initialspikes[0, 2, 0, 1] = 1
    #initialspikes[0, 3, 0, 1] = 1
    initialspikes[0, 4, 0, 1] = 1
    initialspikes[0, 5, 0, 1] = 1
    initialspikes[0, 6, 0, 1] = 1
    initialspikes[0, 7, 0, 1] = 1
    return tf.Variable(tf.cast(initialspikes, tf.dtypes.int32), trainable=False)

  @tf.function
  def __call__(self, log=False):
    if log:
        tf.print(self.population.connections, summarize=-1, sep=',', output_stream='file://data/fullconnections.csv')

    i = tf.constant(0)
    while i < self.iterations:
      self.spikevalues.assign(self.population())
      if log:
        tf.print(self.spikevalues, summarize=-1, sep=',', output_stream='file://data/fullspike.csv')
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

#tf.debugging.set_log_device_placement(True)

iterationCount = 640
population_model = PopulationModule(layer_size=640, iterations=iterationCount, thickness=32, name="populations")
#c = population_model()
#print(c)
#tf.debugging.set_log_device_placement(False)

startTime = datetime.now()
population_model(log=True)
endTime = datetime.now()
duration = endTime - startTime
print(f'Run time for {iterationCount} iterations: {duration} or {duration.seconds / iterationCount} s/iter')