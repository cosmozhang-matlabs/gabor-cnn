import numpy as np
import scipy as sp
import scipy.io as sio
import math
import sys,os
import cv2
import random

projpath = os.path.join( os.path.dirname(os.path.realpath(__file__)), '..' )

s1l2 = sio.loadmat(projpath + '/datas/S1L2/Rsp_tPointsFit.mat')['Rsp_tPointsFit']
s1l3 = sio.loadmat(projpath + '/datas/S1L3/Rsp_tPointsFit.mat')['Rsp_tPointsFit']
s2l2 = sio.loadmat(projpath + '/datas/S2L2/Rsp_tPointsFit.mat')['Rsp_tPointsFit']
s2l3 = sio.loadmat(projpath + '/datas/S2L3/Rsp_RetPointsFit.mat')['Rsp_RetPointsFit']

def parseCoors(cctotal, imsize):
  cctotal = cctotal.reshape(np.max(cctotal.shape))
  nn = cctotal.shape[0]
  coors = [None for i in xrange(nn)]
  for i in xrange(nn):
    cc = cctotal[i]
    cc = cc.reshape(np.max(cc.shape)).astype(np.uint32)
    cy = (cc-1) / imsize
    cx = cc - cy*imsize
    coors[i] = np.stack([cx,cy],axis=1)
  return coors

def calcCenters(coors):
  nn = len(coors)
  centers = np.zeros([nn,2]).astype(np.float32)
  for i in xrange(nn):
    coor = coors[i].astype(np.float32)
    center = np.mean(coor, axis=0)
    centers[i,:] = center
  return centers

def calcDistances(centers):
  nn = centers.shape[0]
  centers = centers.astype(np.float32)
  dists = np.zeros([nn,nn]).astype(np.float32)
  for i in xrange(nn):
    cc = centers[i,:].reshape([2])
    dx = centers[:,0].reshape([nn]) - cc[0]
    dy = centers[:,1].reshape([nn]) - cc[1]
    dists[i,:] = np.sqrt(dx*dx + dy*dy)
  return dists

s1l2_dists = calcDistances( calcCenters( parseCoors( sio.loadmat(projpath + '/datas/S1L2/CCtotal.mat')['CCtotal'], 512 ) ) )
s1l3_dists = calcDistances( calcCenters( parseCoors( sio.loadmat(projpath + '/datas/S1L3/CCtotal.mat')['CCtotal'], 512 ) ) )
s2l2_dists = calcDistances( calcCenters( parseCoors( sio.loadmat(projpath + '/datas/S2L2/CCtotal.mat')['CCtotal'], 512 ) ) )
s2l3_dists = calcDistances( calcCenters( parseCoors( sio.loadmat(projpath + '/datas/S2L3/CCtotal.mat')['CCtotal'], 512 ) ) )

ppd = 15.7 # pixels per degree

def spatial_frequency(i):
  return np.power(2,-2+i) / ppd

def orientation(i):
  return -45 + i*15

def deg2rad(deg):
  return np.pi * deg / 180.0

nsf = 6 # number of spatial frequencies
nod = 12 # number of orientation degrees
sf = spatial_frequency(np.arange(0,nsf).astype('float32')) # spatial frequencies
od = orientation(np.arange(0,nod).astype('float32')) # orientation degrees

def gen_stimuli(size, sf, od):
  if size % 2 != 0:
    raise "Size must be an even number!"
  coorsx = np.repeat(np.arange(size).reshape(1,size)-size/2, size, axis=0).astype(np.float32)
  coorsy = np.repeat(np.arange(size).reshape(size,1)-size/2, size, axis=1).astype(np.float32)
  k = sf
  odrad = deg2rad(od)
  # im = np.cos( k * coorsx * np.cos(odrad) + k * coorsy * np.sin(odrad) )
  im = np.cos( k * 2 * (np.pi) * coorsx * np.cos(odrad) + k * 2 * (np.pi) * coorsy * np.sin(odrad) )
  return im

def sti2image(sti):
  return ((sti + 1) / 2 * 255).astype(np.uint8)
 
ssti = 512 # size of stimuli image
stis = np.zeros([nsf, nod, ssti, ssti]) # stimuli images
for i in range(nsf):
  for j in range(nod):
    stis[i,j,:,:] = gen_stimuli(ssti, sf[i], od[j])

class DataBatch:
  def __init__(self):
    self.x = None
    self.y = None

class Dataset:
  def __init__(self, rsp, dists):
    self.nneurons = rsp.shape[0]
    self.rsp = rsp
    self.dists = dists

  def next_batch(self, batch_size):
    x = np.zeros([batch_size,ssti,ssti,1])
    y = np.zeros([batch_size,self.nneurons])
    for n in xrange(batch_size):
      i = int(random.random() * nsf)
      j = int(random.random() * nod)
      x[n,:,:,0] = gen_stimuli(ssti, sf[i], od[j])
      y[n,:] = self.rsp[:,i,j]
    batch = DataBatch()
    batch.x = x
    batch.y = y
    return batch


if __name__ == "__main__":
  print "s1l2\n", s1l2
  print "s1l3\n", s1l3
  print "s2l2\n", s2l2
  print "s2l3\n", s2l3
  print "s1l2_dists\n", s1l2_dists
  print "s1l3_dists\n", s1l3_dists
  print "s2l2_dists\n", s2l2_dists
  print "s2l3_dists\n", s2l3_dists
  print "sf\n", sf
  # print "od\n", od
  for i in range(nsf):
    for j in range(nod):
      impath = projpath + "/stiimages/1st_sf%d_od%d.tiff" % (i,j)
      cv2.imwrite(impath, sti2image(stis[i,j,:,:].reshape(ssti,ssti)))
