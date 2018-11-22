#!/usr/bin/env ipython
import tensorflow as tf 
    
def load_settings_from_file():
    
    config = {

    "dir_root":"experiments/aegan_sine/",
    "num_samples":1000,
    "seq_length": 1400,
    "num_signals":1,
    "channels":1,
    "latent_dim": 100,
    "batch_size":200,
    "num_epochs":1500,
    "D_rounds":3,
    "G_rounds":5,
    "num_iter_per_epoch": 100000, 
    "beta1" : 0.5,
    "beta2" : 0.9, 
    "generator_settings": { "dim": 8,
                                "filter_size": (5,1),
                                "stride": (2,1),
                                "padding":"SAME",
                                "learning_rate": 0.0002
                           }
        ,
    "discriminator_settings": { "dim": 8,
                                "filter_size": (5,1),
                                "stride": (2,1),
                                "padding":"SAME",
                                "learning_rate": 0.0002
                               },
                "ae_settings": { "dim": 8,
                                "filter_size": (5,1),
                                "stride": (2,1),
                                "padding":"SAME",
                                "learning_rate": 0.0002
                               }

    }    
    
    return config 

