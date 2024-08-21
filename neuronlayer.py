import tensorflow as tf
import numpy as np
from datetime import datetime
from datetime import datetime


class LayerModule(tf.Module):
  def __init__(self, layer_size, thickness=1, name=None):
    super().__init__(name=name)
    self.is_built = False
    self.layer_size = layer_size
    self.thickness = thickness
    self.tflayer_size = tf.constant(layer_size, dtype=tf.int32)
    factor = np.ones([self.thickness, 1, self.layer_size], dtype=np.int32)
    factor = factor * 2
    self.factor = tf.constant(factor, dtype=tf.dtypes.int32)

  def __call__(self):
    # Create variables on first call.
    if not self.is_built:
      layer = self.InitializeConnections()
      self.connections = tf.Variable(layer, name='connections', trainable=False)
      self.potentials = tf.Variable(tf.zeros([self.thickness, 1, self.layer_size], dtype=tf.dtypes.int32), dtype=tf.dtypes.int32, name='potentials', trainable=False)
      initialspikes = np.zeros((self.thickness, 1, self.layer_size), dtype=np.int32)
      initialspikes[0, 0, 1] = 1
      initialspikes[1, 0, 1] = 1
      initialspikes[2, 0, 1] = 1
      initialspikes[3, 0, 1] = 1
      initialspikes[4, 0, 1] = 1
      initialspikes[5, 0, 1] = 1
      initialspikes[6, 0, 1] = 1
      initialspikes[7, 0, 1] = 1
      self.spikes = tf.Variable(tf.cast(initialspikes, tf.dtypes.int32), trainable=False)

      self.is_built = True

    self.potentials.assign((self.spikes @ self.connections) + self.potentials)
    self.spikes.assign(tf.cast(tf.greater(self.potentials, 5), tf.int32))
    self.potentials.assign(tf.cast(tf.divide(self.potentials, self.factor), dtype=tf.dtypes.int32))
    return self.spikes

  def InitializeConnections(self):
      layer = tf.cast(tf.random.normal([self.thickness, self.layer_size, self.layer_size], mean=0.0, stddev=10.0), tf.dtypes.int32)
      """
      layer = np.zeros((self.thickness, self.layer_size, self.layer_size))
      for i in range(self.layer_size - 2):
        layer[0][i][i+1] = 6
        layer[0][i][i+2] = 5
      layer = tf.cast(layer, tf.dtypes.int32)
      """
      return layer

class PopulationModule(tf.Module):
  def __init__(self, layer_size, iterations, thickness=1, name=None):
    super().__init__(name=name)
    self.layer_size = layer_size
    self.iterations = tf.constant(iterations)
    self.thickness = tf.constant(thickness)
    thicks = []
    for pop in range(layer_size):
      thicks.append(f'file://data/population{pop}.csv')
    self.thickname = tf.Variable(thicks, dtype=tf.dtypes.string, name='populationnames', trainable=False)

    self.population1 = LayerModule(layer_size=self.layer_size, thickness=self.thickness, name="pop1")
    self.population2 = LayerModule(layer_size=self.layer_size, thickness=self.thickness, name="pop2")
    self.spikevalues1 = tf.Variable(tf.zeros([self.thickness, 1, self.layer_size], dtype=tf.dtypes.int32), trainable=False)
    self.spikevalues2 = tf.Variable(tf.zeros([self.thickness, 1, self.layer_size], dtype=tf.dtypes.int32), trainable=False)

  @tf.function
  def __call__(self, log=False):
    i = tf.constant(0)
    while i < self.iterations:
      self.spikevalues1.assign(self.population1())
      self.spikevalues2.assign(self.population2())
      if log:
        tf.print(self.spikevalues1, summarize=-1, sep=',', output_stream='file://data/fullspike1.csv')
        """
        pop = tf.constant(0)
        while pop < self.thickness:
          tf.print('Logging population', pop, self.thickname[pop])
          poplayer = self.spikevalues1[0][0]
          popname = self.thickname.numpy()
          tf.print('File name ' + popname)
          tf.print(poplayer, summarize=-1, sep=',', output_stream='file://data/help.csv')
          pop = pop + 1
       """
      #for spike in self.spikevalues1[0][0]:
      #  tf.print(spike, end=',', output_stream='file://data/population0.csv')
      #tf.print(output_stream='file://data/population0.csv')
      #self.population1()
      #self.population2()
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
population_model = PopulationModule(layer_size=320, iterations=iterationCount, thickness=32, name="populations")
#c = population_model()
#print(c)
#tf.debugging.set_log_device_placement(False)

startTime = datetime.now()
population_model(log=True)
endTime = datetime.now()
duration = endTime - startTime
print(f'Run time for {iterationCount} iterations: {duration} or {duration.seconds / iterationCount} s/iter')