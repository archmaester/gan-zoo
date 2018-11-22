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
        self.lambda_p = config['lambda_p']
        self.lambda_r = config['lambda_r']    
        self.generator_settings = config["generator_settings"]
        self.discriminator_settings = config["discriminator_settings"]       
        self.r_settings = config["r_settings"] 
        self.lr = config['learning_rate']
        self.beta1 = config['beta1']
        self.beta2 = config['beta2']
        self.feature_dim = 5600.0
                       
        self.optimize_d = tf.train.AdamOptimizer(0.0002, beta1 = self.beta1, beta2 = self.beta2)
        self.optimize_g = tf.train.AdamOptimizer(0.0002, beta1 = self.beta1, beta2 = self.beta2)
        self.optimize_r = tf.train.AdamOptimizer(0.00002, beta1 = self.beta1, beta2 = self.beta2)
 
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
        
        
    def _encoder(self, x, scope = None, reuse = False):

        with tf.variable_scope(scope, reuse=reuse):
            
            with tf.variable_scope('Encoder', reuse= reuse):
               
                a = self.r_settings["dim"]
                b = self.r_settings["filter_size"]
                c = self.r_settings["stride"]
                d = self.r_settings["padding"]
                
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

                x = tf.layers.dense(x, 88 * 1 * a * 4)
                x = tf.layers.batch_normalization(x)
                x = tf.nn.relu(x)

                print("Decoder : ", x.get_shape())  

                x = tf.reshape(x, [-1, 88 , 1, a * 4])

                print("Decoder : ", x.get_shape())

                x = tf.layers.conv2d_transpose(x, a * 2, b, c, d)
                x = tf.layers.batch_normalization(x)
                x = tf.nn.relu(x)
                
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
#                 x = tf.nn.sigmoid(x)                          
                
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
            
            return out, x, feature                 
        
    def build_gan(self):
        
        self.ze_input = self._encoder(self.x_input, scope = 'AutoEncoder')
        print("ZE:", self.ze_input.get_shape())
        self.xr_input = self._decoder(self.ze_input, scope = 'AutoEncoder')
        print("XR:", self.xr_input.get_shape())
        self.xg_input = self._decoder(self.z_input, scope = 'AutoEncoder', reuse = True)
        print("XG:", self.xg_input.get_shape())
        self.D_real,  self.D_real_logits,  self.f_x_input    = self._discriminator(self.x_input, reuse = False)
        self.D_fake,  self.D_fake_logits,  self.f_xg_input   = self._discriminator(self.xg_input, reuse = True)
        self.D_recon, self.D_recon_logits, self.f_xr_input   = self._discriminator(self.xr_input, reuse = True)
        
        # Compute gradient penalty
        epsilon = tf.random_uniform(shape=[tf.shape(self.x_input)[0],1, 1, 1], minval=0., maxval=1.)
        print("Epsilon:", epsilon.get_shape())
        self.interpolation = epsilon * self.x_input + (1 - epsilon) * self.xg_input
        print("Interpolation :", self.interpolation.get_shape())
        _, self.d_inter, _ = self._discriminator(self.interpolation, reuse=True)
        print("D_inter:", self.d_inter.get_shape())
        
    def _calculate_loss(self):
        
        with tf.variable_scope('loss'):


            self.lambda_w =tf.sqrt(self.z_dim / self.feature_dim )
            self.reconstruction   = tf.reduce_mean(tf.nn.l2_loss(self.f_x_input - self.f_xr_input)) #reconstruction

            # Doubtful

            gradients = tf.gradients([self.d_inter], [self.interpolation])[0]
            slopes = tf.sqrt(tf.reduce_mean(tf.square(gradients), reduction_indices=[1]))
            penalty = tf.reduce_mean((slopes - 1) ** 2)

            self.md_x       = tf.reduce_mean(self.f_xr_input - self.f_xg_input)
            self.md_z       = tf.reduce_mean(self.ze_input - self.z_input) * self.lambda_w
            self.reg     = tf.square(self.md_x - self.md_z)


            self.d_real_loss  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_logits, labels=tf.ones_like(self.D_real)))
            self.d_fake_loss  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.zeros_like(self.D_fake)))
            self.d_recon_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_recon_logits, labels=tf.ones_like(self.D_recon)))
            
            self.D_loss  = 0.95 * self.d_real_loss + 0.05 * self.d_recon_loss + self.d_fake_loss + self.lambda_p * penalty
            
            self.R_loss  = self.reconstruction + self.lambda_r * self.reg
            
            self.G_loss  = tf.abs(tf.reduce_mean(self.D_real) - tf.reduce_mean(self.D_fake))

        
            print(self.D_loss.get_shape())
            print(self.R_loss.get_shape())
            print(self.G_loss.get_shape())
            
           
    def initialize_optimizer(self):

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('Discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('AutoEncoder/Decoder')]
        r_vars = [var for var in t_vars if var.name.startswith('AutoEncoder')]
        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

            self.d_opt = self.optimize_d.minimize(self.D_loss, var_list = d_vars)
            self.g_opt = self.optimize_g.minimize(self.G_loss, var_list = g_vars)
            self.r_opt = self.optimize_r.minimize(self.R_loss, var_list = r_vars)
             
    def init_saver(self):
        
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver()
        