# Copyright 2018 Srikanth Gadicherla @imsrgadich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import tensorflow as tf
import numpy as np
from gpflow.decors import params_as_tensors,autoflow
from gpflow.params import Parameter
from gpflow.kernels import Kernel
from gpflow import transforms,settings

import pdb

class SpectralMixture(Kernel):
    def __init__(self, num_mixtures=1, mixture_weights=[],\
                 mixture_scales=[],mixture_means=[],\
                 input_dim=1,active_dims=None,name='SpectralMixture'):
        '''
        - num_mixtures is the number of mixtures; denoted as Q in
        Wilson 2013.
        - mixture_weights
        - mixture_variance is 
        - mixture_means is the list (or array) of means of the 
        mixtures.
        - input_dim is the dimension of the input to the kernel.
        - active_dims is the dimension of the X which needs to be used.
        
        References:  
        http://hips.seas.harvard.edu/files/wilson-extrapolation-icml-2013_0.pdf
        http://www.cs.cmu.edu/~andrewgw/typo.pdf
        
        '''
        super().__init__(input_dim,active_dims,name=name)
        # Q(num_of_mixtures)=1 then SM kernel is SE Kernel.
        if num_mixtures == 1:
            print("Using default mixture = 1")
        
        # number of mixtures is non trainable.
        self.num_mixtures = num_mixtures 
        #Parameter(num_mixtures,trainable=False)
        
        self.mixture_weights = mixture_weights
        
        self.mixture_scales = mixture_scales
        
        self.mixutre_means = mixture_means
    
    
    #@autoflow((tf.float32,),(tf.float32,))
    #@params_as_tensors
    def initialize_(self, train_x, train_y):
        '''
        TODO: how to use autoflow here. 
        explicitly converting to tensor for debugging purposes
        '''
#        train_x = tf.convert_to_tensor(train_x,dtype=tf.float32)
#        train_y = tf.convert_to_tensor(train_y,dtype=tf.float32)
        
        self.input_dim = np.shape(train_x)[1]
        
        if np.size(train_x.shape) == 1:
            train_x = np.expand_dims(train_x,-1)
        if np.size(train_x.shape) == 2:
            train_x = np.expand_dims(train_x,0)
        
        train_x_sort = np.copy(train_x)
        train_x_sort.sort(axis=1)

        max_dist = np.squeeze(train_x_sort[:,-1, :] - train_x_sort[:,0, :])
        
        min_dist_sort = np.squeeze(np.abs(train_x_sort[:,1:, :] - train_x_sort[:,:-1, :]))
        min_dist = np.zeros([self.input_dim],dtype=float)

        # min of each data column could be zero. Hence, picking minimum which is not zero
        for ind in np.arange(self.input_dim):
            min_dist[ind] = min_dist_sort[np.amin(np.where(min_dist_sort[:,ind]>0),axis=1),ind]
        
        # for random restarts during batch processing. We need to initialize at every 
        # batch. Lock the seed here.
        seed= np.random.randint(low=1,high=10**10)
        np.random.seed(seed)
        
        #Inverse of lengthscales should be drawn from truncated Gaussian |N(0, max_dist^2)|
        # dim: Q x D
        #self.mixture_scales = tf.multiply(,tf.cast(max_dist,dtype=tf.float32)**(-1)
        
        self.mixture_scales = (np.multiply(np.abs(np.random.randn(self.num_mixtures,\
                      self.input_dim)),np.expand_dims(max_dist,axis=0)))**(-1)
        self.mixture_scales = Parameter(self.mixture_scales,\
                                        transform=transforms.positive)
        
        # Draw means from Unif(0, 0.5 / minimum distance between two points), dim: Q x D
        # the nyquist is half of maximum frequency. TODO
        nyquist = np.divide(0.5,min_dist)
        self.mixture_means = np.multiply(np.random.rand(self.num_mixtures\
                            ,self.input_dim),np.expand_dims(nyquist,0))
        self.mixutre_means = Parameter(self.mixture_means)
        
        # Mixture weights should be roughly the std of the y values divided by 
        # the number of mixtures
        # dim: 1 x Q
        self.mixture_weights= np.divide(np.std(train_y,axis=0),\
                            self.num_mixtures)*np.ones(self.num_mixtures)
        self.mixture_weights= Parameter(self.mixture_weights)
        pdb.set_trace()
        return None
   
    @params_as_tensors
    def K(self, X1, X2=None):
        if self.mixture_weights == [] or self.mixture_means == [] \
                                      or self.mixture_scales == []:
                raise RuntimeError('Parameters of spectral mixture kernel not initialized.\
                                    Run `sm_kern_object.initialize_(train_x,train_y)`.')
                # initialization can only be done by user as it needs target data as well.        
        if X2 is None:
            X2 = X1
        
        X1 = tf.transpose(tf.expand_dims(X1,-1),perm=[1,2,0])
                                                         #D x 1 x N1
        X2 = tf.expand_dims(tf.transpose(X2,perm=[1,0]),-1)#D x N2 x 1
        
        t = tf.abs(tf.subtract(X1,X2)) # D x N2 x N1
        
        
        # we will optimize the standard deviations.
        
        for ind in tf.range(self.input_dim):
            # write the exp term with looping over dimesion and
            # tf.reduce_prod(tf.multiply(t,tf.transpose(self.dim[ind,:],perm=[1 0])),axis=0) #take product over dimension
            # should give N x N matrix
            # 

        exp_term = tf.multiply(tf.square(tf.matmul(t,\
                                self.mixture_scales)),-2.*math.pi**2)
        cos_term = tf.multiply(tf.square(tf.matmul(t,\
                                self.mixture_means)),2.*math.pi)
        res = tf.squeeze(tf.reduce_prod(tf.multiply(tf.exp(exp_term),\
                                        tf.cos(cos_term)),axis=0))
        res = tf.squeeze(tf.reduce_sum(tf.multiply(res,\
                                        self.mixture_weights,axis=0)))
        return t
    
    @params_as_tensors
    def Kdiag(self, X):
        
        # just the sum of weights. Weights represent the signal
        # variance. 
        return tf.fill(tf.stack([tf.shape(X)[0]]),\
                              tf.sum(self.mixture_weights))

