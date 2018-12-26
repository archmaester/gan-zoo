import numpy as np
from operator import itemgetter 
import random
from math import ceil

class Data:
    
    def __init__(self, config):

        self.num_samples = config['num_samples']
        self.seq_length = config['seq_length']
        self.num_signals = config['num_signals']
        
        self.sine_wave()
        self.samples = {}
        self.labels = {}
        
        self.split(proportions = [0.6,0.2,0.2], random_seed = 43)
        self.__print_shapes__()
        
        
    def next_batch_train(self, batch_size):
        
        idx = np.random.choice(len(self.samples['train']), batch_size)
        
        yield itemgetter(*idx)(self.samples['train']), itemgetter(*idx)(self.labels['train'])
        
        
    def sine_wave(self, freq_low=1, freq_high=5, amplitude_low = 0.1, amplitude_high=0.9, **kwargs):
        
        ix = np.arange(self.seq_length) + 1
        samples = []
        labels = []
        for i in range(self.num_samples):
            signals = []
            for i in range(self.num_signals):
                f = np.random.uniform(low=freq_high, high=freq_low)     # frequency
                A = np.random.uniform(low=amplitude_high, high=amplitude_low)        # amplitude
                # offset
                offset = np.random.uniform(low=-np.pi, high=np.pi)
                signals.append(A*np.sin(2*np.pi*f*ix/float(self.seq_length) + offset))
            samples.append(np.array(signals).T)
            labels.append(0)
        # the shape of the samples is num_samples x seq_length x num_signals
        samples = np.asarray(samples)
        
        samples = np.expand_dims(samples, axis=2)

        self.samples_all = samples
        self.labels_all = labels
        
    def split(self, proportions, random_seed= None):
        """
        Return train/validation/test split.
        """

        if random_seed != None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        assert np.sum(proportions) == 1

        n_total = self.samples_all.shape[0]
        n_train = ceil(n_total*proportions[0])
        n_test = ceil(n_total*proportions[2])
        n_vali = n_total - (n_train + n_test)

        # permutation to shuffle the samples

        shuff = np.random.permutation(n_total)
        train_indices = shuff[:n_train]
        vali_indices = shuff[n_train:(n_train + n_vali)]
        test_indices = shuff[(n_train + n_vali):]

        # split up the samples
        train = self.samples_all[train_indices]
        vali = self.samples_all[vali_indices]
        test = self.samples_all[test_indices]

        if self.labels_all != None:

            train_labels = itemgetter(*train_indices)(self.labels_all)
            vali_labels = itemgetter(*vali_indices)(self.labels_all)
            test_labels = itemgetter(*test_indices)(self.labels_all)
            labels_split = [train_labels, vali_labels, test_labels]

        else:

            labels_split = None 

        self.samples['train'] = train
        self.samples['vali'] = vali
        self.samples['test'] = test 
        
        self.labels['train'],self.labels['vali'], self.labels['test'] = labels_split
        
    def __print_shapes__(self):
        
        
        print("Shape of Train data : x, y ",self.samples['train'].shape,",",len(self.labels['train']))
        print("Shape of Valid data : x, y ",self.samples['vali'].shape,",",len(self.labels['vali']))
        print("Shape of Test data  : x, y ",self.samples['test'].shape,",",len(self.labels['test']))
        
        