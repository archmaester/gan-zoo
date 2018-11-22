#!/usr/bin/env ipython
import json
    
def load_settings_from_file(identifier):
    
    config = {

    "dir_root":"experiments/sine_aegon/",
    "num_samples":1000,
    "seq_length": 40,
    "num_signals":1,
    "channels":1,
    "latent_dim": 10,
    "batch_size":20,
    "num_epochs":1500,
    "D_rounds":3,
    "G_rounds":5,
    "num_iter_per_epoch": 1000, 
    "generator_settings": { "n_filters":[8, 1],
                            "filter_sizes":[(3,1), (3,1), (3,1)],
                            "strides": [(2,1), (2,1), (2,1)],
                            "activations": [tf.nn.relu, tf.nn.relu, tf.nn.relu],
                            "paddings":["SAME", "SAME", "SAME", "SAME"],
                            "beta": 0.5 
                           },
    "discriminator_settings": { "n_filters":[8, 1],
                                "filter_sizes":[(3,1), (3,1), (3,1)],
                                "strides": [(2,1), (2,1), (2,1)],
                                "activations": [None, None, None],
                                "paddings":["SAME", "SAME", "SAME", "SAME"],
                                "learning_rate": 0.0009,
                                "beta": 0.5 
                               },
    "activations_enc": [None, None, None, None, None, None]
    }    
    
    return config 

