import re
import os
import sys
from neuronconfiguration import NeuronConfiguration
from initloader import InitLoader
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
      parts = re.split(fileparse, entry.name)
      if parts[1] == 'simulation':
        sims.append(int(parts[2]))

  return max(sims) + 1

def MakeSimulationFolder(simulationNumber):
  foldername = path + basefoldername + str(simulationNumber)
  os.makedirs(foldername, exist_ok=True)

  return foldername


class LayerModule(tf.Module):
  """
  This class extends the Tensorflow Module class, so that any method decorated
  with the @tf.function notation will be compiled into a compute graph, on first
  execution, and all subsequent iterations will run on the compute device.
  The functor of this class implements a single tick of the spiking neural algorithm,
  including learning.
  """

  def __init__(self, configuration: NeuronConfiguration, init_loader: InitLoader, name=None):
    super().__init__(name=name)
    self.is_built = False

    self.configuration = configuration
    self.init_loader = init_loader

    self.iterations = self.configuration.GetIterationCount()
    self.layer_size = self.configuration.GetLayerSize()
    self.thickness = self.configuration.GetThickness()
    self.outputwidth = self.configuration.GetOutputWidth()
    self.interconnectCount = configuration.GetInterconnectCount()
    self.inputwidth = self.outputwidth * self.interconnectCount
    self.interconnectpad = tf.zeros((self.thickness, 1, self.layer_size-self.inputwidth), dtype=tf.int32)
    spiketrain = tf.Variable(self.init_loader.GenerateSpikes(self.iterations), trainable=False)
    self.spiketrain = spiketrain
    self.tick = tf.Variable(0)
    self.tflayer_size = tf.constant(self.layer_size, dtype=tf.int32)
    factor = np.ones([self.thickness, 1, self.layer_size], dtype=np.int32)
    factor = factor * 2
    self.factor = tf.constant(factor, dtype=tf.dtypes.int32)
    self.threshold = tf.constant(12)
    self.hebblearning = tf.constant(2)
    hebbdelay = np.ones([self.thickness, 1, self.layer_size], dtype=np.int32)
    hebbdelay = hebbdelay * 7
    self.hebbdelay = tf.constant(hebbdelay, dtype=tf.dtypes.int32)
    self.activehebbbase = tf.zeros([self.thickness, self.layer_size, self.layer_size], dtype=tf.dtypes.int32)
    self.connections = tf.Variable(self.init_loader.InitializeConnections(), name='connections', trainable=False)
    self.connection_delays = tf.Variable(tf.ones((self.thickness, self.layer_size, self.layer_size), dtype=tf.int32), name='connection_delays', trainable=False)
    self.connection_timers = tf.Variable(tf.zeros((self.thickness, self.layer_size, self.layer_size), dtype=tf.int32), name='connection_timers', trainable=False)
    self.connection_post_timers = tf.Variable(tf.zeros((self.thickness, self.layer_size, self.layer_size), dtype=tf.int32), name='connection_post_timers', trainable=False)
    self.post_time_delay = tf.constant(25)

    self.interconnects = tf.constant(self.init_loader.InitializeInterconnects(), name='interconnects')
    self.delaytimes = tf.Variable(tf.zeros([self.thickness, 1, self.layer_size], dtype=tf.dtypes.int32), dtype=tf.dtypes.int32, name='delaytimes', trainable=False)
    self.delayguards = tf.Variable(tf.ones([self.thickness, 1, self.layer_size], dtype=tf.dtypes.int32), dtype=tf.dtypes.int32, name='delayguards', trainable=False)

  def Interconnect(self):
    """ Each population has an entry in self.interconnects that describes the populations whose outputs feed into its inputs.
        Outputs are collected from the first self.outputwidth cells of each interconnecting population (0 - self.outputwidth-1), 
        and stacked in the order specified by self.interconnects.  

        Example: if self.outputwidth = 2 and population 20 has populations 19 and 21 
        (in that order) listed in self.interconnects, then the current spiking values of [19(0), 19(1), 21(0), 21(1)]
        will be stacked in that order and the resulting four spike values will be injected into the last four cells of population
        20: [20(-4), 20(-3), 20(-2), 20(-1)].
    """
    sout = tf.slice(self.spikes, begin=[0, 0, 0], size=[self.thickness, 1, self.outputwidth])
    sin = tf.reshape(tf.gather_nd(indices=self.interconnects, params=sout), [self.thickness, 1, self.inputwidth])
    interconnect_spikes = tf.concat([self.interconnectpad, sin], axis=2)
    #tf.concat([tf.gather_nd(sout, indices=[[0],[2],[1],[3]]), pad], axis=2)
    return interconnect_spikes

  def DelayConnect(self):
    activedelays = (self.activehebbbase + tf.transpose(self.spikes, perm=[0,2,1])) * self.connection_delays
    self.connection_timers.assign_add(activedelays)
    triggeredtimers = tf.cast(tf.equal(self.connection_timers, 1), tf.int32)
    activeconnections = triggeredtimers * self.connections
    self.potentials.assign((self.dummyspikes @ activeconnections) + self.decayedpotentials)
    self.connection_timers.assign(tf.maximum(self.connection_timers - 1, 0))

    activeconnectionmask = tf.cast(tf.greater(activeconnections, 0), tf.int32)
    self.connection_post_timers.assign_add(activeconnectionmask * self.post_time_delay)
    self.connection_post_timers.assign(tf.maximum(self.connection_post_timers - 1, 0))

  def HebbLearningConnect(self):
    # Broadcast hebbtimers across all rows of a workspace, then filter out columns that are not spiking this tick.
    activehebb = (self.activehebbbase + tf.minimum(tf.transpose(self.hebbtimers, perm=[0,2,1]), self.hebblearning)) * self.spikes

    # Add resulting columns container hebbtimers to existing connections, capping any connection at the spike threshold.
    # Remember to exclude any connections that are not already above zero.
    activehebb = tf.multiply(tf.cast(tf.greater_equal(self.connections, 0), tf.int32), activehebb)
    self.connections.assign(tf.cast(tf.minimum(self.connections + activehebb, self.threshold), tf.int32))

    self.hebbtimers.assign_add((self.spikes * self.hebbdelay))
    self.hebbtimers.assign(tf.cast(tf.maximum(tf.subtract(self.hebbtimers, 1), 0), tf.int32))

  def HebbLearning(self):
    # Broadcast hebbtimers across all rows of a workspace, then filter out columns that are not spiking this tick.
    activehebb = (self.activehebbbase + tf.minimum(tf.transpose(self.hebbtimers, perm=[0,2,1]), self.hebblearning)) * self.spikes

    # Add resulting columns container hebbtimers to existing connections, capping any connection at the spike threshold.
    # Remember to exclude any connections that are not already above zero.
    activehebb = tf.multiply(tf.cast(tf.greater(self.connections, 0), tf.int32), activehebb)
    self.connections.assign(tf.cast(tf.minimum(self.connections + activehebb, self.threshold), tf.int32))

    self.hebbtimers.assign_add((self.spikes * self.hebbdelay))
    self.hebbtimers.assign(tf.cast(tf.maximum(tf.subtract(self.hebbtimers, 1), 0), tf.int32))

  def HebbLearningEager(self):
    # Broadcast hebbtimers across all rows of a workspace, then filter out columns that are not spiking this tick.
    activehebb = (self.activehebbbase + tf.minimum(tf.transpose(self.hebbtimers, perm=[0,2,1]), self.hebblearning)) * self.spikes

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
      self.dummyspikes = tf.Variable(tf.ones([self.thickness, 1, self.layer_size], dtype=tf.int32), name='dummy_spikes', trainable=False)

      self.is_built = True


    # potential(j) += SUM(ij)[spike(i) @ connection(ij)]
    self.DelayConnect()
    #self.potentials.assign((self.spikes @ self.connections) + self.decayedpotentials)

    # Do learning while self.spikes contains the presynaptic spike pattern.
    self.HebbLearning()

    # Spike if above threshold, but only if delaytime is exhausted.  This generates the post-synaptic spike pattern.
    self.spikes.assign(tf.cast(tf.greater_equal(tf.multiply(self.potentials, self.delayguards), self.threshold), tf.int32))

    # Inject any spikes included in the incoming spike train plus interconnects.
    self.spikes.assign(tf.cast(tf.minimum(self.spikes + self.spiketrain[self.tick] + self.Interconnect(), 1), tf.int32))
    self.tick.assign_add(1)

    # delaytime(i) = delaytime(i) + 8 if spike(i) else delaytime(i)
    self.delaytimes.assign_add(tf.multiply(self.spikes, 8))

    # delayguards can be used to filter out cells with delaytime > 0.  Delayguard is 1 if delaytime <= 0.
    self.delayguards.assign(tf.cast(tf.less_equal(self.delaytimes, 0), tf.int32))

    # decaypotential(i) = potential(i) / 2, or 0 if delaytime is delaying.
    self.decayedpotentials.assign(tf.cast(tf.divide(tf.multiply(self.potentials, self.delayguards), self.factor), dtype=tf.dtypes.int32))

    self.delaytimes.assign(tf.cast(tf.maximum(tf.subtract(self.delaytimes, 1), 0), tf.int32))

    if log:
      #tf.print(self.connections, summarize=-1, sep=',', output_stream= 'file://' + datafolder + 'fullconnections.dat')
      tf.print(self.spikes, summarize=-1, sep=',', output_stream= 'file://' + datafolder + 'fullspike.dat')
      tf.print(self.potentials, summarize=-1, sep=',', output_stream= 'file://' + datafolder + 'fullactivations.dat')
      tf.print(self.hebbtimers, summarize=-1, sep=',', output_stream= 'file://' + datafolder + 'fullhebbtimers.dat')

    return self.spikes
  


class PopulationModule(tf.Module):
  """
  This class extends the Tensorflow Module class, so that any method decorated
  with the @tf.function notation will be compiled into a compute graph, on first
  execution, and all subsequent iterations will run on the compute device.
  The functor of this class will call the worker class LayerModule the correct
  number of times, as indicated by the configuration.
  """

  def __init__(self, configuration: NeuronConfiguration, init_loader: InitLoader, name=None):
    super().__init__(name=name)

    self.configuration = configuration
    self.init_loader = init_loader
    self.layer_size = self.configuration.GetLayerSize()
    iterations = self.configuration.GetIterationCount()
    self.iterations = tf.constant(iterations)
    thickness = self.configuration.GetThickness()
    self.thickness = tf.constant(thickness)

    self.population = LayerModule(configuration=self.configuration, init_loader=self.init_loader, name="population")

    # This Tensorflow variable may not be necessary, but seems to be required to ensure that calls to self.population
    # are included in the compute graph.  TBD
    self.spikevalues = tf.Variable(tf.zeros([self.thickness, 1, self.layer_size], dtype=tf.dtypes.int32), trainable=False)


  @tf.function
  def __call__(self, datafolder, log=False):
    """
    Functor that turns an instance of this class into a callable function.  Since it is decorated
    with the Tensorflow @tf.function notation, the code here plus methods it calls will be compiled
    into a compute graph on first execution.
    """
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


def Run(configuration: NeuronConfiguration):
  """
  Run the simulation described by the given configuration.
  """
  #tf.debugging.set_log_device_placement(True)

  simulationNumber = GetNextSimulationNumber()
  datafolder = MakeSimulationFolder(simulationNumber) + '/'
  configuration.Save(datafolder)

  layerSize = configuration.GetLayerSize()
  thickness = configuration.GetThickness()
  iterationCount = configuration.GetIterationCount()

  print(f'Running simulation {simulationNumber} with layer size {layerSize}, thickness {thickness}, iteration count {iterationCount}')

  initializers = configuration.GetInitializers()
  selected_initializer = configuration.GetSelectedInitializer()
  print(f'Initializers are {initializers}, using initializer {selected_initializer}')
  init_loader = InitLoader(initializers[selected_initializer], configuration)

  population_model = PopulationModule(configuration, init_loader, name="populations")
  #c = population_model()
  #print(c)
  #tf.debugging.set_log_device_placement(False)

  startTime = datetime.now()
  population_model(datafolder=datafolder, log=True)
  endTime = datetime.now()
  duration = endTime - startTime
  print(f'Run time for {iterationCount} iterations: {duration} or {duration.seconds / iterationCount} s/iter')

  with DataPrep(simulationNumber) as prep:
    #prep.BuildConnections()
    prep.BuildSpikes()
    prep.BuildActivations()
    prep.BuildHebbianTimers()

# Execution starts here.
if __name__ == "__main__":
  if len(sys.argv) < 2:
    print(f'Usage: {sys.argv[0]} <configuration> [initializer number] [iterations] [layersize] [thickness]')
    exit(0)

  configuration = NeuronConfiguration(sys.argv[1])
  if not configuration.valid:
    print(f'Configuration {sys.argv[1]} is not valid')
    exit(0)

  if len(sys.argv) > 2:
    initializer = int(sys.argv[2])
    if initializer >= len(configuration.GetInitializers()):
      print(f'Initializer {initializer} is bigger than allowed by configuration {sys.argv[1]}, which has {len(configuration.GetInitializers())} initializers')
      exit(0)

    configuration.SetSelectedInitializer(initializer)

  if len(sys.argv) > 3:
    configuration.SetIterationCount(int(sys.argv[3]))

  if len(sys.argv) > 4:
    configuration.SetIterationCount(int(sys.argv[4]))

  if len(sys.argv) > 5:
    configuration.SetLayerSize(int(sys.argv[5]))

  if len(sys.argv) > 6:
    configuration.SetThickness(int(sys.argv[6]))

  Run(configuration)
