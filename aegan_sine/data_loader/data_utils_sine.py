import numpy as np
from operator import itemgetter 

class Data:
    
    def __init__(self, config):

        self.num_samples = config['num_samples']
        self.seq_length = config['seq_length']
        self.num_signals = config['num_signals']
        
        self.sine_wave()
        
    def next_batch(self, batch_size):
        
        idx = np.random.choice(self.num_samples, batch_size)
        
        yield itemgetter(*idx)(self.samples), itemgetter(*idx)(self.labels)
        
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

        self.samples = samples
        self.labels = labels
        