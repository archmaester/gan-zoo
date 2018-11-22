import tensorflow as tf
import ast
from models.model_base import BaseModel 

class Model(BaseModel):
    
    def __init__(self, config):
        
        super(Model, self).__init__(config)

        self.z_dim = config['latent_dim']
        self.seq_length = config['seq_length']
        self.num_signals = config['num_signals']
        self.channels = config['channels']

        self.generator_settings = config["generator_settings"]
        self.discriminator_settings = config["discriminator_settings"]       
        self.r_settings = config["r_settings"] 

        self.beta1 = config['beta1']
        self.beta2 = config['beta2']
               
        self.optimize_d = tf.train.AdamOptimizer(0.0002, beta1 = self.beta1, beta2 = self.beta2)
        self.optimize_g = tf.train.AdamOptimizer(0.0002, beta1 = self.beta1, beta2 = self.beta2)
        self.optimize_r = tf.train.AdamOptimizer(0.0002, beta1 = self.beta1, beta2 = self.beta2)
 
        self.build_model()
        self.init_saver()
        
        
    def _add_placeholders(self):
           
        self.z_input = tf.placeholder(tf.float32, shape=[None, self.z_dim, 1, 1])
        self.x_input = tf.placeholder(tf.float32, shape=[None, self.seq_length, self.num_signals, self.channels])
   
    def build_model(self):
        
        self._add_placeholders()
        self.build_gan()
        self._calculate_loss()
        self.initialize_optimizer()
        
        
    def _encoder(self, x, scope = None, reuse = False):

        with tf.variable_scope(scope, reuse=reuse):
            
            with tf.variable_scope('Encoder', reuse= reuse):
            
                print("Encoder_Input:", x.get_shape())

                for (a,b,c,d,e) in zip(self.r_settings["n_filters"], self.r_settings["filter_sizes"], self.r_settings["strides"], self.r_settings["paddings"], self.r_settings["activations"]):
                    
                    if not reuse:
                        print("Encoder_CNN : ", x.get_shape())
                    
                    x = tf.layers.conv2d(x, a, b, c , d, activation = e)

                return tf.layers.batch_normalization(x) 
                
                
    def _decoder(self, x, scope = None, reuse = False):
  
        with tf.variable_scope(scope, reuse= reuse):
        
            with tf.variable_scope('Decoder', reuse= reuse):
    

                for (a,b,c,d,e) in zip(self.generator_settings["n_filters"], self.generator_settings["filter_sizes"], self.generator_settings["strides"], self.generator_settings["paddings"], self.generator_settings["activations"]):

                    print("Decoder : ", x.get_shape())

                    x = tf.layers.conv2d_transpose(x, a, b, c, d, activation = e)
                    
                print("Decoder : ", x.get_shape())
                
                
                return tf.nn.tanh(x)
            
    def _map_random_z_input_to_encoder_z_input(self, x,  reuse = False):
        
        with tf.variable_scope('Mapping', reuse=reuse):
            
            x = tf.layers.flatten(x)
            
            x = tf.layers.dense(x, self.z_dim, activation = tf.nn.tanh)
            
            x = tf.reshape(x, shape = [-1, self.z_dim, 1, 1])
            
            return x 
                    
    def _discriminator(self, x, reuse = False):
        
        with tf.variable_scope('Discriminator', reuse=reuse):
            
            print("Discriminator_Input:", x.get_shape())
        
            for (a,b,c,d,e) in zip(self.discriminator_settings["n_filters"], self.discriminator_settings["filter_sizes"], self.discriminator_settings["strides"], self.discriminator_settings["paddings"], self.discriminator_settings["activations"]):

                x = tf.layers.conv2d(x, a, b, c , d, activation = e)
                
                if not reuse:
                    print("Discriminator_CNN : ", x.get_shape())
            
            x = tf.layers.flatten(x)
            tf.layers.dense(x,1)

            out = tf.nn.sigmoid(x)
            
            return out, x                 
        
    def build_gan(self):
        
        self.ze_input = self._encoder(self.x_input, scope = 'AutoEncoder')
        self.xr_input = self._decoder(self.ze_input, scope = 'AutoEncoder')
        self.za_input = self._map_random_z_input_to_encoder_z_input(self.z_input)
        
        self.D_real, self.D_real_logits = self._discriminator(self.x_input, reuse = False)
        self.D_fake, self.D_fake_logits = self._discriminator(self.xr_input, reuse = True)
        
    def _calculate_loss(self):
        
        with tf.variable_scope('loss'):
            
            # Reconstruction Loss
            self.R_loss   = tf.reduce_mean(tf.nn.l2_loss(self.x_input - self.xr_input)) #reconstruction Loss

            # Discriminator Loss
            self.D_real_loss  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_logits, labels=tf.ones_like(self.D_real)))
            self.D_fake_loss  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.zeros_like(self.D_fake)))
            self.D_loss = self.D_real_loss + self.D_fake_loss

            # Generator Loss 
            self.G_loss  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.ones_like(self.D_fake)))
            
            
    def initialize_optimizer(self):

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('Discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('AutoEncoder/Decoder')]
        r_vars = [var for var in t_vars if var.name.startswith('Mapping')]
        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

            self.d_opt = self.optimize_d.minimize(self.D_loss, var_list = d_vars)
            self.g_opt = self.optimize_g.minimize(self.G_loss, var_list = g_vars)
            self.r_opt = self.optimize_r.minimize(self.R_loss, var_list = r_vars)
            
             
    def init_saver(self):
        
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver()
        