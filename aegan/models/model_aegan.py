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
        self.ae_settings = config["ae_settings"] 
        self.optimize_d = tf.train.AdamOptimizer(self.discriminator_settings['learning_rate'], beta1 =  self.discriminator_settings['beta'])
        self.optimize_g = tf.train.AdamOptimizer(self.generator_settings['learning_rate'], beta1 = self.generator_settings['beta'])
        self.optimize_ae = tf.train.AdamOptimizer(self.ae_settings['learning_rate'], beta1 = self.ae_settings['beta'])
 
        self.build_model()
        self.init_saver()
        
        
    def _add_placeholders(self):
           
        self.z_input = tf.placeholder(tf.float32, shape=[None, self.z_dim, 1, 1])
        self.x_input = tf.placeholder(tf.float32, shape=[None, self.seq_length, self.num_signals, self.channels])
        self.t = tf.placeholder(tf.float32, shape=())
   
    def build_model(self):
        
        self._add_placeholders()
        self.build_gan()
        self._calculate_loss()
        self.initialize_optimizer()
        
        
    def _encoder(self, scope = None, reuse = False):

        with tf.variable_scope(scope, reuse=reuse):
            
            with tf.variable_scope('Encoder', reuse= reuse):
            
                x = self.x_input 

                print("Encoder_Input:", x.get_shape())

                for (a,b,c,d,e) in zip(self.ae_settings["n_filters"], self.ae_settings["filter_sizes"], self.ae_settings["strides"], self.discriminator_settings["paddings"], self.ae_settings["activations"]):

                    
                    if not reuse:
                        print("Encoder_CNN : ", x.get_shape())
                    
                    x = tf.layers.conv2d(x, a, b, c , d, activation = e)

#                 x = tf.layers.flatten(x)
#                 x = tf.layers.dense(x,175)
                
#                 x = tf.expand_dims(x, axis =-1)
#                 x = tf.expand_dims(x, axis =-1)

                self.enc_out = tf.layers.batch_normalization(x) 
                
                
    def _decoder(self, scope = None, reuse = False):
  
        with tf.variable_scope(scope, reuse= reuse):
        
            with tf.variable_scope('Decoder', reuse= reuse):
        
                x = self.dec_input

                for (a,b,c,d,e) in zip(self.generator_settings["n_filters"], self.generator_settings["filter_sizes"], self.generator_settings["strides"], self.generator_settings["paddings"], self.generator_settings["activations"]):

                    print("Decoder : ", x.get_shape())

                    x = tf.layers.conv2d_transpose(x, a, b, c, d, activation = e)
                    
                print("Decoder : ", x.get_shape())
                
                
                self.gen_sample = tf.nn.tanh(x)  
                    
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
        
        self._encoder(scope = 'AutoEncoder')
        self.dec_input = tf.add(tf.scalar_mul((self.t), self.enc_out), tf.scalar_mul((1 - self.t) , self.z_input))
        print(self.enc_out.get_shape())
        print(self.z_input.get_shape())
        print(self.dec_input.get_shape())

        self._decoder(scope = 'AutoEncoder')
        self.D_real, self.D_real_logits = self._discriminator(self.x_input, reuse = False)
        self.D_fake, self.D_fake_logits = self._discriminator(self.gen_sample, reuse = True)
        
    def _calculate_loss(self):
        
        with tf.variable_scope('loss'):
            
            self.G_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.ones_like(self.D_fake)))
            self.D_loss_fake = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.zeros_like(self.D_fake)))            
            self.D_loss_real = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_logits, labels=tf.ones_like(self.D_real)))    
            
            self.D_loss = self.D_loss_real + self.D_loss_fake 
            
            self.l2 = self.x_input - self.gen_sample
            self.ae_loss = tf.nn.l2_loss(self.l2)
            self.ae_loss = tf.reduce_mean(self.ae_loss)

            self.ae_loss = tf.scalar_mul((self.t), self.ae_loss)
            self.G_loss = tf.scalar_mul((1 - self.t) , self.G_loss)
            
           
    def initialize_optimizer(self):

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('Discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('AutoEncoder/Decoder')]
        ae_vars = [var for var in t_vars if var.name.startswith('AutoEncoder')]
        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

            self.d_opt = self.optimize_d.minimize(self.D_loss, var_list = d_vars)
            self.g_opt = self.optimize_g.minimize(self.G_loss, var_list = g_vars)
            self.ae_opt = self.optimize_ae.minimize(self.ae_loss, var_list = ae_vars)
             
    def init_saver(self):
        
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver()
        