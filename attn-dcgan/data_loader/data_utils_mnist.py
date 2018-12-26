import numpy as np
from operator import itemgetter 
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib
matplotlib.use('agg')
import pylab as plt
class Data:
    
    def __init__(self, config):

        self.num_samples = config['num_samples']
        self.seq_length = config['seq_length']
        self.num_signals = config['num_signals']
        
        source='./data/mnist/'
        
        self.mnist = input_data.read_data_sets(source)
        print(self.mnist)
        
        self.samples = self.mnist.train.images
        self.samples = np.reshape(self.samples,(-1,28,28,1))
        

    def next_batch(self, batch_size):
        
        idx = np.random.choice(self.num_samples, batch_size)
        
        yield itemgetter(*idx)(self.samples)
        
         