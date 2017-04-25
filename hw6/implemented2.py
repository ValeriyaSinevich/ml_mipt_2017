
# coding: utf-8

# In[7]:

from __future__ import print_function
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne


# In[3]:

theano.sandbox.cuda.dnn.version()


# In[5]:



# set up plots

get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt

# reload external libs during development
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

#try:
    #from lasagne.layers.dnn import BatchNormDNNLayer as BatchNormLayer
    #from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
    #from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
    #print ("dnn")
#except ImportError:
from lasagne.layers import BatchNormLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer

# In[6]:

def create_v3(input_var, input_shape=(3, 32, 32),
              ccp_num_filters=[64, 128, 128, 256], ccp_filter_size=3,
              fc_num_units=[256, 256], num_classes=10,
              **junk):
    # input layer
    network = lasagne.layers.InputLayer(shape=input_shape,
                                        input_var=input_var)
    # conv-relu-conv-relu-pool layers
    for num_filters in ccp_num_filters:
        network = Conv2DLayer(
            network, num_filters=num_filters,
            filter_size=(ccp_filter_size, ccp_filter_size),
            pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(gain='relu'))
        network = Conv2DLayer(
            network, num_filters=num_filters,
            filter_size=(ccp_filter_size, ccp_filter_size),
            pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(gain='relu'))
        network = MaxPool2DLayer(network, pool_size=(2, 2))
    # fc-relu
    for num_units in fc_num_units:
        network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.2),
            num_units=num_units,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(gain='relu'))
    # output layer
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.2),
        num_units=num_classes,
        nonlinearity=lasagne.nonlinearities.softmax)
    return network


# In[ ]:


