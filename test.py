import re
import os
import sys
import math
from neuronconfiguration import NeuronConfiguration
from initloader import InitLoader
import tensorflow as tf
import numpy as np
from datetime import datetime
from dataprep import DataPrep

c = NeuronConfiguration('sequence_1')

class Test(tf.Module):
  def __init__(self, configuration: NeuronConfiguration, init_loader: InitLoader, name=None):
    super().__init__(name=name)
    self.is_built = False

    self.layer_size = 16
    self.thickness = 4
    self.outputwidth = 2
    self.interconnectCount = 4
    self.threshold = 12
    self.inputwidth = self.outputwidth * self.interconnectCount
    self.xedgesize, self.yedgesize = self.GenerateSizes()

    self.activehebbbase = tf.zeros([self.thickness, self.layer_size, self.layer_size], dtype=tf.dtypes.int32)
    self.hebbtimers = tf.Variable(tf.zeros([self.thickness, 1, self.layer_size], dtype=tf.dtypes.int32), dtype=tf.dtypes.int32, name='hebbtimers', trainable=False)
    self.hebblearning = tf.constant(2)
    hebbdelay = np.ones([self.thickness, 1, self.layer_size], dtype=np.int32)
    hebbdelay = hebbdelay * 7
    self.hebbdelay = tf.constant(hebbdelay, dtype=tf.dtypes.int32)
    self.connection_delays = tf.Variable(self.InitializeConnectionDelays(), name='connection_delays', trainable=False)
    self.connection_timers = tf.Variable(tf.zeros((self.thickness, self.layer_size, self.layer_size), dtype=tf.int32), name='connection_timers', trainable=False)
    self.connection_post_timers = tf.Variable(tf.zeros((self.thickness, self.layer_size, self.layer_size), dtype=tf.int32), name='connection_post_timers', trainable=False)
    self.post_time_delay = tf.constant(25)
    self.spikes = tf.Variable(self.InitializeSpikes(), trainable=False)
    self.dummyspikes = tf.Variable(tf.ones([self.thickness, 1, self.layer_size], dtype=tf.int32), name='dummy_spikes', trainable=False)
    self.connections = tf.Variable(self.InitializeConnections(), name='connections', trainable=False)
    self.potentials = tf.Variable(tf.zeros([self.thickness, 1, self.layer_size], dtype=tf.dtypes.int32), dtype=tf.dtypes.int32, name='potentials', trainable=False)
    self.decayedpotentials = tf.Variable(tf.zeros([self.thickness, 1, self.layer_size], dtype=tf.dtypes.int32), dtype=tf.dtypes.int32, name='decayedpotentials', trainable=False)

    self.delaytimes = tf.Variable(tf.zeros([self.thickness, 1, self.layer_size], dtype=tf.dtypes.int32), dtype=tf.dtypes.int32, name='delaytimes', trainable=False)
    self.delayguards = tf.Variable(tf.ones([self.thickness, 1, self.layer_size], dtype=tf.dtypes.int32), dtype=tf.dtypes.int32, name='delayguards', trainable=False)


  def GenerateSizes(self):
    edgesize = int(math.sqrt(self.layer_size))
    xedgesize = edgesize
    yedgesize = edgesize
    # If not a perfect square, use the smallest rectangle that fully contains all cells.
    while xedgesize * yedgesize < self.layer_size:
      yedgesize += 1

    return (xedgesize, yedgesize)

  def InitializeSpikes(self):
    initialspikes = np.zeros((self.thickness, 1, self.layer_size), dtype=np.int32)
    for population in range(self.thickness):
      for cell in range(0, self.layer_size, 4):
        initialspikes[population, 0, cell] = 1

    return initialspikes

  def InitializeConnectionDelays(self):
    initialdelays = np.ones((self.thickness, self.layer_size, self.layer_size), dtype=np.int32)
    """
    for population in range(self.thickness):
      for row in range(self.layer_size):
        for cell in range(self.layer_size):
          initialdelays[population, row, cell] = population+row+cell
    """
    return initialdelays


  def InitializeConnections(self):
    """ Set the internal connections between neurons in each of the single populations.
    """
    layer = np.zeros((self.thickness, self.layer_size, self.layer_size), dtype=np.int32)


    for population in range(self.thickness):
      for ycell in range(self.yedgesize):
        out_cell = ycell*self.xedgesize + self.yedgesize - 1
        for xcell in range(self.xedgesize-2):
            from_cell = ycell*self.xedgesize + xcell
            layer[population][from_cell][from_cell+1] = self.threshold+1
            layer[population][from_cell][out_cell] = 1

    return layer

  @tf.function
  def DelayConnect(self):
    activedelays = (self.activehebbbase + tf.transpose(self.spikes, perm=[0,2,1])) * self.connection_delays
    self.connection_timers.assign_add(activedelays)
    triggeredtimers = tf.cast(tf.equal(self.connection_timers, 1), tf.int32)
    activeconnections = triggeredtimers * self.connections
    activepotentials = self.dummyspikes @ activeconnections
    self.potentials.assign(activepotentials + self.decayedpotentials)
    self.connection_timers.assign(tf.maximum(self.connection_timers - 1, 0))
    activeconnectionmask = tf.cast(tf.greater(activeconnections, 0), tf.int32)
    self.connection_post_timers.assign_add(activeconnectionmask * self.post_time_delay)
    self.connection_post_timers.assign(tf.maximum(self.connection_post_timers - 1, 0))

    active_postsynaptic = tf.cast(tf.greater(activepotentials, 0), tf.int32)
    delay_correction = (self.activehebbbase + self.post_time_delay) * activeconnectionmask
    self.connection_delays.assign_add(delay_correction - ((self.activehebbbase + active_postsynaptic) * self.connection_post_timers))
    #self.connection_delays.assign_add(triggeredtimers * self.connection_post_timers)

  @tf.function
  def HebbLearning(self):
    # Broadcast hebbtimers across all rows of a workspace, then filter out columns that are not spiking this tick.
    activehebb = (self.activehebbbase + tf.minimum(tf.transpose(self.hebbtimers, perm=[0,2,1]), self.hebblearning)) * self.spikes

    # Add resulting columns container hebbtimers to existing connections, capping any connection at the spike threshold.
    # Remember to exclude any connections that are not already above zero.
    activehebb = tf.multiply(tf.cast(tf.greater(self.connections, 0), tf.int32), activehebb)
    self.connections.assign(tf.cast(tf.minimum(self.connections + activehebb, self.threshold), tf.int32))

    self.hebbtimers.assign_add((self.spikes * self.hebbdelay))
    self.hebbtimers.assign(tf.cast(tf.maximum(tf.subtract(self.hebbtimers, 1), 0), tf.int32))

