import tensorflow as tf
import ast
from models.model_base import BaseModel 
import numpy as np 

class Model(BaseModel):
    
    def __init__(self, config):
        
        super(Model, self).__init__(config)

        self.z_dim = config['latent_dim']
        self.seq_length = config['seq_length']
        self.num_signals = config['num_signals']
        self.channels = config['channels']
        self.generator_settings = config["generator_settings"]
        self.discriminator_settings = config["discriminator_settings"]              
        self.ae_settings = config["ae_settings"]
        
        self.beta1 = config['beta1']
        self.beta2 = config['beta2']
                       
        self.optimize_d = tf.train.AdamOptimizer(self.discriminator_settings['learning_rate'], beta1 = self.beta1, beta2 = self.beta2)
        self.optimize_g = tf.train.AdamOptimizer(self.generator_settings['learning_rate'],     beta1 = self.beta1, beta2 = self.beta2)
        self.optimize_r = tf.train.AdamOptimizer(self.ae_settings['learning_rate'],            beta1 = self.beta1, beta2 = self.beta2)
 
        self.build_model()
        self.init_saver()
        
        
    def _add_placeholders(self):
           
        self.z_input = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.x_input = tf.placeholder(tf.float32, shape=[None, self.seq_length, self.num_signals, self.channels])
        self.t = tf.placeholder(tf.float32, shape=())
        
    def build_model(self):
        
        self._add_placeholders()
        self.build_gan()
        self._calculate_loss()
        self.initialize_optimizer()                                    
                
    def _encoder(self, x, scope = None, reuse = False):

        with tf.variable_scope(scope, reuse=reuse):
            
            with tf.variable_scope('Encoder', reuse= reuse):
               
                a = self.ae_settings["dim"]
                b = self.ae_settings["filter_size"]
                c = self.ae_settings["stride"]
                d = self.ae_settings["padding"]
                
                print("Encoder:", x.get_shape())
                
                x = tf.layers.conv2d(x, a*1, b, c, d)
                x = tf.nn.relu(x)

                print("Encoder:", x.get_shape())
                
                x = tf.layers.conv2d(x, a*2, b, c, d)
                x = tf.layers.batch_normalization(x)
                x = tf.nn.relu(x)

                print("Encoder:", x.get_shape())
                
                x = tf.layers.conv2d(x, a*4, b, c, d)
                x = tf.layers.batch_normalization(x)
                x = tf.nn.relu(x)

                print("Encoder:", x.get_shape())
                
                x = tf.layers.flatten(x)
                x = tf.layers.dense(x, self.z_dim)

                print("Encoder:", x.get_shape())

                return x 
                                    
                
    def _decoder(self, x, scope = None, reuse = False):
  
        with tf.variable_scope(scope, reuse= reuse):
        
            with tf.variable_scope('Decoder', reuse= reuse):
                
                a = self.generator_settings["dim"]
                b = self.generator_settings["filter_size"]
                c = self.generator_settings["stride"]
                d = self.generator_settings["padding"]

                print("Decoder : ", x.get_shape())  

                x = tf.layers.dense(x, 4 *4 * a * 4)
                x = tf.layers.batch_normalization(x)
                x = tf.nn.relu(x)

                print("Decoder : ", x.get_shape())  

                x = tf.reshape(x, [-1, 4, 4, a * 4])

                print("Decoder : ", x.get_shape())

                x = tf.layers.conv2d_transpose(x, a *2, b, c, d)
                x = tf.layers.batch_normalization(x)
                x = tf.nn.relu(x)
                
                print("Decoder : ", x.get_shape())  

                x = tf.layers.flatten(x)

                print("Decoder : ", x.get_shape())  

                x = tf.layers.dense(x, 7 *7 * a * 2)
                
                print("Decoder : ", x.get_shape())  

                x = tf.reshape(x, [-1, 7, 7, a * 2])

                print("Decoder : ", x.get_shape())  

                x = tf.layers.conv2d_transpose(x, a *1, b, c, d)
                x = tf.layers.batch_normalization(x)
                x = tf.nn.relu(x)

                print("Decoder : ", x.get_shape())  

                x = tf.layers.conv2d_transpose(x, self.channels, b, c, d)

                print("Decoder : ", x.get_shape())  
                x = tf.nn.sigmoid(x)                          
                
                return x
                    
    def _discriminator(self, x, reuse = False):
        
        with tf.variable_scope('Discriminator', reuse=reuse):
            
            a = self.generator_settings["dim"]
            b = self.discriminator_settings["filter_size"]
            c = self.discriminator_settings["stride"]
            d = self.discriminator_settings["padding"]

            print("Discriminator:", x.get_shape())

            x = tf.layers.conv2d(x, a*1, b, c, d)
            x = tf.nn.leaky_relu(x)

            print("Discriminator:", x.get_shape())

            x = tf.layers.conv2d(x, a*2, b, c, d)
            x = tf.layers.batch_normalization(x)
            x = tf.nn.leaky_relu(x)

            print("Discriminator:", x.get_shape())

            x = tf.layers.conv2d(x, a*4, b, c, d)
            x = tf.layers.batch_normalization(x)
            x = tf.nn.leaky_relu(x)  

            print("Discriminator:", x.get_shape()) 

            x = tf.layers.flatten(x) 

            print("Discriminator:", x.get_shape()) 

            feature = x 

            x = tf.layers.dense(x, 1)

            out = tf.nn.sigmoid(x)
            
            return out, x                
              
        
    def build_gan(self):
        
        self.ze_input = self._encoder(self.x_input, scope = 'AutoEncoder')
        
        self.dec_input = tf.add(tf.scalar_mul((self.t), self.ze_input), tf.scalar_mul((1 - self.t) , self.z_input))
        self.xg_input = self._decoder(self.dec_input, scope = 'AutoEncoder')
        
        print("XG:", self.xg_input.get_shape())
        
        self.D_real,  self.D_real_logits  = self._discriminator(self.x_input, reuse = False)
        self.D_fake,  self.D_fake_logits  = self._discriminator(self.xg_input, reuse = True)
     
    
    def _calculate_loss(self):
        
        with tf.variable_scope('loss'):


            self.d_real_loss  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_logits, labels=tf.ones_like(self.D_real)))
            self.d_fake_loss  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.zeros_like(self.D_fake)))
            self.G_loss  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.ones_like(self.D_fake)))
            
            self.l2 = self.x_input - self.xg_sample
            self.ae_loss = tf.nn.l2_loss(self.l2)
            self.ae_loss = tf.reduce_mean(self.ae_loss)

            self.ae_loss = tf.scalar_mul((self.t), self.ae_loss)
            self.G_loss = tf.scalar_mul((1 - self.t) , self.G_loss)
            
            self.D_loss  = self.d_real_loss + self.d_fake_loss
                   
    def initialize_optimizer(self):

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('Discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('Decoder')]
        ae_vars = [var for var in t_vars if var.name.startswith('AutoEncoder')]


        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

            self.d_opt = self.optimize_d.minimize(self.D_loss, var_list = d_vars)
            self.g_opt = self.optimize_g.minimize(self.G_loss, var_list = g_vars)
            self.ae_opt = self.optimize_ae.minimize(self.ae_loss, var_list = ae_vars)
             
    def init_saver(self):
        
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver()
        