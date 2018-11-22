import tensorflow as tf

from data_loader.data_utils_sine import Data
from models.model_aegan import Model
from trainers.trainer import Trainer
from settings.config_aegan import load_settings_from_file
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.plot import Plot
import os 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

session_conf = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
session_conf.gpu_options.allow_growth = True

def main():
    
    # PROLOG
    settings = load_settings_from_file()
    
    # Create directories
    create_dirs(settings['dir_root'])
        
    # Create Tensorflow Session
    sess = tf.Session(config = session_conf)
    
    # Create the data generator
    data = Data(settings)
    
    # Create Model 
    model = Model(settings)
    
    # Create tensorflow Logging 
    logger = Logger(sess, settings)
    
    #Creating plots
    plot = Plot(sess, settings)
    
    sess.run(tf.global_variables_initializer())

    # Create trainer object
    trainer = Trainer(sess, model, data, settings, logger, plot)
    
    # Train the model 
    trainer.train_epoch()

    
if __name__ == '__main__':
    
    main()