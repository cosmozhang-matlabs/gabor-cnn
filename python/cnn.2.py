import tensorflow as tf
import numpy as np
import datas
import sys,os
import time

ssti = datas.ssti
nn = datas.s1l2.shape[0]
nnn = 16
indexes = np.arange(nn)
np.random.shuffle(indexes)
indexes = indexes[:nnn]
# dataset = datas.Dataset(datas.s1l2[indexes,:,:], datas.s1l2_dists[indexes,:][:,indexes])
dataset = datas.Dataset(datas.s1l2, datas.s1l2_dists)

def weight_variable(shape, stddev=0.1):
  initial = tf.truncated_normal(shape, stddev=stddev)
  return tf.Variable(initial)

def bias_variable(shape, val=0.0):
  initial = tf.constant(val, shape=shape)
  return tf.Variable(initial)

def gabor_kernel(kSize, kChanIn, kChanOut):
  sigs = tf.truncated_normal([kChanIn, kChanOut], stddev=0.1) + 1
  sfs = tf.random_uniform([kChanIn, kChanOut], minval=0.2, maxval=8)
  ods = tf.random_uniform([kChanIn, kChanOut], minval=-np.pi, maxval=np.pi)
  sigs = tf.Variable(sigs)
  sfs = tf.Variable(sfs)
  ods = tf.Variable(ods)
  coorsx = np.repeat(np.arange(kSize).reshape(1,kSize)-kSize/2, kSize, axis=0).astype(np.float32)
  coorsy = np.repeat(np.arange(kSize).reshape(kSize,1)-kSize/2, kSize, axis=1).astype(np.float32)
  maps = [[None for i in range(kChanIn)] for j in range(kChanOut)]
  for i in range(kChanOut):
    for j in range(kChanIn):
      s = sigs[j,i]
      f = sfs[j,i]
      o = ods[j,i]
      coorsx_ = coorsx * tf.cos(o) + coorsy * tf.sin(o)
      s2 = s*s
      maps[i][j] = 1 / (2*np.pi*s2) * tf.exp(-(coorsx*coorsx+coorsy*coorsy)/s2) * tf.cos(f*coorsx_)
  stacked = tf.stack([tf.stack(maps[i],axis=2) for i in range(kChanOut)], axis=3)
  return stacked

def recurrent_kernel(distance_matrix):
  nn = distance_matrix.shape[0]
  amp1 = tf.Variable(tf.truncated_normal([nn], stddev=0.1) + 1)
  amp2 = tf.Variable(tf.truncated_normal([nn], stddev=0.1) + 1)
  sig1 = tf.Variable(tf.truncated_normal([nn], stddev=0.1) + 1)
  sig2 = tf.Variable(tf.truncated_normal([nn], stddev=0.1) + 1)
  coorsx = np.repeat(np.arange(nn).reshape(1,nn)-nn/2, nn, axis=0).astype(np.float32)
  coorsy = np.repeat(np.arange(nn).reshape(nn,1)-nn/2, nn, axis=1).astype(np.float32)
  maps = [None for i in xrange(nn)]
  for i in xrange(nn):
    a1 = amp1[i]
    a2 = amp2[i]
    s1 = sig1[i]
    s2 = sig2[i]
    s12 = s1 * s1
    s22 = s2 * s2
    disrow = distance_matrix[i,:].reshape([nn])
    gauss1 = a1 / (2*np.pi*s12) * tf.exp( - disrow*disrow / (2*s1*s1) )
    gauss2 = a2 / (2*np.pi*s22) * tf.exp( - disrow*disrow / (2*s2*s2) )
    maps[i] = gauss1 - gauss2
  stacked = tf.stack(maps, axis=1)
  return stacked

# train

class GaborCNN:
  def __init__(self):
    self.saver = None
    self.nneurons = dataset.nneurons
    self.x = tf.placeholder(tf.float32, [None, datas.ssti, datas.ssti, 1])
    # self.x = tf.placeholder(tf.float32, [None, 784])
    self.y = tf.placeholder(tf.float32, [None, self.nneurons])
    self.dists = dataset.dists
    self.keep_prob = tf.placeholder(tf.float32)
    self.yPredition = None
    self.variables = []
    self.layers = []
    self.outs = {}
    self.init = None
    self.loss = None
    self.train = None
    self.accuracy = None
    self.sess = None
    self.ts = 0

  def timeInit(self):
    self.ts = time.time()

  def timeElapsed(self, msg):
    msg += " "
    while len(msg) < 50:
      msg = msg + "-"
    print "%s | elapsed(s): %.3f" % (msg, time.time() - self.ts)

  def construct(self, convs=[(17,36)], recurrents=1):
    y = self.x
    self.outs = {}
    self.layers = [y]
    self.outs["y0"] = y
    self.variables = []

    cnt = 1

    # convolution
    lastChan = y.shape.as_list()[3]
    for convConfig in convs:
      kSize = convConfig[0]
      kChan = convConfig[1]
      convKernel = gabor_kernel(kSize, lastChan, kChan)
      lastChan = kChan
      self.timeElapsed("gabor_kernel ok")
      convBias = bias_variable([kChan])
      y = tf.nn.conv2d( input = y, filter = convKernel, strides = [1,1,1,1], padding = "VALID" ) + convBias
      self.layers.append(y)
      self.outs["y%dc"%cnt] = y
      self.variables.append(convKernel)
      self.outs["w%d"%cnt] = convKernel
      self.variables.append(convBias)
      self.outs["b%d"%cnt] = convBias
      cnt += 1
    yShape = y.shape.as_list()
    poolSize = yShape[1]
    y = tf.nn.max_pool( value = y, ksize = [1,poolSize,poolSize,1], strides = [1,1,1,1], padding = "VALID" )
    self.layers.append(y)
    self.outs["y%dp"%cnt] = y
    cnt += 1

    # here y's shape should be [batch, 1, 1, last_channels]
    yShape = y.shape.as_list()
    # reshape y to vector
    y = tf.reshape(y, [-1,yShape[1]*yShape[2]*yShape[3]])
    yShape = y.shape.as_list()

    # fc: from limited channels to all neurons
    W = weight_variable([y.shape.as_list()[1],self.nneurons])
    b = bias_variable([self.nneurons])
    y = tf.matmul(y,W) + b
    self.layers.append(y)
    self.layers.append(y)
    self.outs["y_s2a"] = y
    self.variables.append(W)
    self.outs["w_s2a"] = W
    self.variables.append(b)
    self.outs["b_s2a"] = b
    cnt += 1

    # full-connected layers
    lastNeuronNum = y.shape.as_list()[1]
    recurrentKernel = recurrent_kernel(self.dists)
    self.timeElapsed("recurrent_kernel ok")
    self.variables.append(recurrentKernel)
    for i in xrange(recurrents):
      y = tf.matmul(y,recurrentKernel)
      self.layers.append(y)
      self.outs["y%d"%cnt] = y
      cnt += 1
    self.timeElapsed("recurrent_kernels added")

    # dropout layer
    y = tf.nn.dropout(y, self.keep_prob)
    self.layers.append(y)
    self.outs["y_dropout"] = y
    cnt += 1

    # output layer (softmax)
    self.outs["y_out"] = y
    self.yPredition = y

  def evaluate(self):
    # self.loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.yPredition) )
    self.loss = tf.reduce_mean( tf.reduce_sum( (self.y-self.yPredition)*(self.y-self.yPredition), axis = 1 ), axis = 0 )
    self.timeElapsed("Evaluation `loss` prepared")
    self.variables.append(self.loss)
    self.timeElapsed("Evaluation `loss` added")
    self.train = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)
    self.timeElapsed("Optimizer prepared")
    # correct_prediction = tf.equal( tf.sigmoid(self.yPredition) > 0.5, self.y > 0.5 )
    # self.accuracy = tf.reduce_mean( tf.cast(correct_prediction, tf.float32) )

  def initialize(self, filename=None):
    self.saver = tf.train.Saver()
    # self.init = tf.variables_initializer(self.variables)
    self.init = tf.initialize_all_variables()
    self.sess = tf.Session()
    self.sess.run(self.init)
    if filename:
      self.saver.restore(self.sess, filename)

  def prepare(self, filename=None):
    self.timeInit()
    self.timeElapsed("Start to prepare network.")
    self.construct()
    self.timeElapsed("Network construction complete.")
    self.evaluate()
    self.timeElapsed("Network evaluations ready.")
    self.initialize(filename)
    self.timeElapsed("Network variables initialized.")

  def train_step(self, batch_size=10, evaluate=False, loss=False):
    new_batch = dataset.next_batch(batch_size)
    batch_xs = new_batch.x
    batch_ys = new_batch.y
    self.sess.run(self.train, feed_dict = {self.x: batch_xs, self.y: batch_ys, self.keep_prob: 0.8})
    # if evaluate:
    #   return self.sess.run([self.accuracy, self.loss], feed_dict = {self.x: batch_xs, self.y: batch_ys, self.keep_prob: 1})

  def test_step(self, batch_size=10):
    # return self.sess.run([self.accuracy, self.loss], feed_dict = {self.x: dataset.test.data, self.y: dataset.test.labels, self.keep_prob: 1})
    new_batch = dataset.next_batch(batch_size)
    batch_xs = new_batch.x
    batch_ys = new_batch.y
    return self.sess.run([self.loss], feed_dict = {self.x: batch_xs, self.y: batch_ys, self.keep_prob: 1})

  def outNames(self):
    return self.outs.keys()

  def readParam(self, paramName):
    val = self.sess.run(self.outs[paramName])
    return val

  def trainEpoches(self, epoches=1):
    batch_size = 100
    batch_num = dataset.train.data.shape[0] / batch_size
    for epoch in range(epoches):
      for i in range(batch_num):
        self.train_step()
    loss = self.test_step()[0]

  def save(self, filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
      os.makedirs(dirname)
    self.saver.save(self.sess, filename)

def prepareNet(filename = None):
  net = GaborCNN()
  net.prepare(filename)
  return net


save_dir = "./train.ckpt"
save_filename = "save.ckpt"
save_path = os.path.join(save_dir,save_filename)
save_epoch_path = save_path + ".epoch"

if __name__ == '__main__':
  if len(sys.argv) == 1:
    tf.device("/gpu:1")
    batch_size = 10
    batch_num = 10
    net = GaborCNN()
    start_epoch = 0
    if os.path.exists(save_dir):
      net.prepare(save_path)
      f = open(save_epoch_path)
      start_epoch = int(f.read())
      f.close()
      print "State loaded from: %s" % save_path
    else:
      net.prepare()
    print "Start training >>>>>>>>>"
    for epoch in range(start_epoch, 1000):
      # test = net.test_step()
      # accuracy = test[0]
      # loss = test[1]
      # print "Epoch %d: accuracy = %f , loss = %f" % (epoch, accuracy, loss)
      test = net.test_step()
      loss = test[0]
      print "Epoch %d: loss = %f" % (epoch, loss)
      if (epoch % 10) == 0:
        net.save(save_path)
        f = open(save_epoch_path, "w")
        f.write("%d"%epoch)
        f.close()
        print "State saved to: %s" % save_path
      for i in range(batch_num):
        net.train_step(batch_size)
  elif sys.argv[1] == "w":
    if os.path.exists(save_dir):
      n = prepareNet(save_path)
      print "Accuracy: %f" % n.test_step()[0]
      print "Params:", n.outNames()
      w1 = n.readParam('w1')
      w2 = n.readParam('w2')
      im = n.sess.run(n.outs['y1c'], feed_dict = {n.x: dataset.test.data[0:1,:,:,:]})
      import scipy.io as sio
      sio.savemat("/home/cosmo/downloads/w.mat", {'w1':w1, 'w2':w2, 'im':im})
    else:
      n = prepareNet()
      w0 = n.readParam('w1')
      print "Accuracy: %f" % n.trainEpoches(5)
      print "Params:", n.outNames()
      w1 = n.readParam('w1')
      w2 = n.readParam('w2')
      im = n.sess.run(n.outs['y1c'], feed_dict = {n.x: dataset.test.data[0,:,:,:]})
      import scipy.io as sio
      sio.savemat("/home/cosmo/downloads/w.mat", {'w0':w0, 'w1':w1, 'w2':w2, 'im':im})