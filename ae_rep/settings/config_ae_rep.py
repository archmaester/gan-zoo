#!/usr/bin/env ipython
import tensorflow as tf 
    
def load_settings_from_file():
    
    config = {

    "dir_root":"experiments/ae_rep/",
    "num_samples":10000,
    "seq_length": 1400,
    "num_signals":1,
    "channels":1,
    "latent_dim": 175,
    "batch_size":200,
    "num_epochs":1500,
    "D_rounds":3,
    "G_rounds":5,
    "num_iter_per_epoch": 10000, 
    "lambda_reg": 1.0,
    "lambda_w": 0.15625,
    "beta1" : 0.5,
    "beta2" : 0.9, 
    "learning_rate" : 0.1,
    "generator_settings": { "n_filters":[32, 16, 8, 1],
                            "filter_sizes":[(3,1), (3,1), (3,1), (3,1)],
                            "strides": [(2,1), (2,1), (2,1), (1,1), (2,1)],
                            "activations": [None, None, None, None],
                            "paddings":["SAME", "SAME", "SAME", "SAME"],
                            "learning_rate": 0.0009,
                            "beta": 0.5 
                           },
    "discriminator_settings": { "n_filters":[32, 64, 128, 1],
                                "filter_sizes":[(3,1), (3,1), (3,1), (3,1)],
                                "strides": [(2,1), (2,1), (2,1), (2,1)],
                                "activations": [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu],
                                "paddings":["SAME", "SAME", "SAME", "SAME"],
                                "learning_rate": 0.00009,
                                "beta": 0.5 
                                   },
                "r_settings": { "n_filters":[16, 32, 64, 1],
                                "filter_sizes":[(3,1), (3,1), (3,1), (3,1)],
                                "strides": [(2,1), (2,1), (2,1), (1,1)],
                                "activations": [None, None, None, None],
                                "paddings":["SAME", "SAME", "SAME", "SAME"],
                                "learning_rate": 0.005,
                                "beta": 0.5 
                               }
    }    
    
    return config 

