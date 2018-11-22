#!/usr/bin/env ipython
import tensorflow as tf 
    
def load_settings_from_file():
    
    config = {

    "dir_root":"experiments/cnn_mnist/",
    "num_samples":10000,
    "seq_length": 28,
    "num_signals":28,
    "channels":1,
    "latent_dim": 100,
    "batch_size":200,
    "num_epochs":1500,
    "D_rounds":3,
    "G_rounds":5,
    "num_iter_per_epoch": 10000, 
    "lambda_p": 1.0,
    "lambda_r": 1.0,
    "beta1" : 0.5,
    "beta2" : 0.9, 
    "learning_rate" : 0.1,
    "generator_settings": { "dim": 64,
                                "filter_size": 5,
                                "stride": 2,
                                "padding":"SAME",
                                "learning_rate": 0.005,
                                "beta": 0.5 
                           }
        ,
    "discriminator_settings": { "dim": 64,
                                "filter_size": 5,
                                "stride": 2,
                                "padding":"SAME",
                                "learning_rate": 0.005,
                                "beta": 0.5 
                               },
                "r_settings": { "dim": 64,
                                "filter_size": 5,
                                "stride": 2,
                                "padding":"SAME",
                                "learning_rate": 0.005,
                                "beta": 0.5 
                               }
    }    
    
    return config 

