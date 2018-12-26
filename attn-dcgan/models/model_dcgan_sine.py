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

        self.beta1 = config['beta1']
        self.beta2 = config['beta2']
                       
        self.optimize_d = tf.train.AdamOptimizer(self.discriminator_settings['learning_rate'], beta1 = self.beta1, beta2 = self.beta2)
        self.optimize_g = tf.train.AdamOptimizer(self.generator_settings['learning_rate'],     beta1 = self.beta1, beta2 = self.beta2)
 
        self.build_model()
        self.init_saver()
        
        
    def _add_placeholders(self):
           
        self.z_input = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.x_input = tf.placeholder(tf.float32, shape=[None, self.seq_length, self.num_signals, self.channels])
    
    def build_model(self):
        
        self._add_placeholders()
        self.build_gan()
        self._calculate_loss()
        self.initialize_optimizer()                                    
                
    def _generator(self, x, reuse = False):
  
        # Generator        
        with tf.variable_scope('Generator', reuse= reuse):

            a = self.generator_settings["dim"]
            b = self.generator_settings["filter_size"]
            c = self.generator_settings["stride"]
            d = self.generator_settings["padding"]

            print("Decoder : ", x.get_shape())  

            x = tf.layers.dense(x, 88 * 1 * a * 4)
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)

            print("Decoder : ", x.get_shape())  

            x = tf.reshape(x, [-1, 88 , 1, a * 4])

            print("Decoder : ", x.get_shape())

            x = tf.layers.conv2d_transpose(x, a * 2, b, c, d)
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)

            x = self.attention(x, a*2)
            
            print("Decoder : ", x.get_shape())  

            x = tf.layers.flatten(x)

            print("Decoder : ", x.get_shape())  

            x = tf.layers.dense(x, 175 * 1 * a * 2)

            print("Decoder : ", x.get_shape())  

            x = tf.reshape(x, [-1, 175, 1, a * 2])

            print("Decoder : ", x.get_shape())  

            x = tf.layers.conv2d_transpose(x, a *1, b, c, d)
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)

            print("Decoder : ", x.get_shape())  

            x = tf.layers.conv2d_transpose(x, a *1, b, c, d)
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)

            print("Decoder : ", x.get_shape())  

            x = tf.layers.conv2d_transpose(x, self.channels, b, c, d)

            print("Decoder : ", x.get_shape())                           

            x = tf.nn.tanh(x) 
            
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

            x = self.attention(x, a*2)
                               
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
    
    def attention(self, x, ch , scope='attention', reuse=False):
        
        with tf.variable_scope(scope, reuse=reuse):
            
            f = tf.layers.conv2d(x, ch // 4, 1, 1) # [bs, h, w, c']
            g = tf.layers.conv2d(x, ch // 4, 1, 1) # [bs, h, w, c']
            h = tf.layers.conv2d(x, ch, 1, 1) # [bs, h, w, c]
            
            # N = h * w
            s = tf.matmul(self.hw_flatten(g), self.hw_flatten(f), transpose_b=True) # # [bs, N, N]

            beta = tf.nn.softmax(s, axis=-1)  # attention map

            o = tf.matmul(beta, self.hw_flatten(h)) # [bs, N, C]
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
            
            o = tf.expand_dims(o,2)
            x = gamma * o + x

        return x
    
    def hw_flatten(self, x) :
        
        return tf.squeeze(x, 2)

    def build_gan(self):
        
        self.xg_input = self._generator(self.z_input)
        print("XG:", self.xg_input.get_shape())
        
        self.D_real,  self.D_real_logits  = self._discriminator(self.x_input, reuse = False)
        self.D_fake,  self.D_fake_logits  = self._discriminator(self.xg_input, reuse = True)
     
    
    def _calculate_loss(self):
        
        with tf.variable_scope('loss'):

            self.d_real_loss  = tf.reduce_mean(tf.nn.relu(1- self.D_real_logits))
            self.d_fake_loss  = tf.reduce_mean(tf.nn.relu(1 + self.D_fake_logits))
            self.G_loss  = -tf.reduce_mean(self.D_fake_logits)
            
            self.D_loss  = self.d_real_loss + self.d_fake_loss
                   
    def initialize_optimizer(self):

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('Discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('Generator')]

        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

            self.d_opt = self.optimize_d.minimize(self.D_loss, var_list = d_vars)
            self.g_opt = self.optimize_g.minimize(self.G_loss, var_list = g_vars)
             
    def init_saver(self):
        
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver()
        