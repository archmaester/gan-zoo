from tqdm import tqdm
import numpy as np
from operator import itemgetter 

class Trainer:
    
    def __init__(self, sess, model, data, settings, logger, plot ):
        
        self.plot = plot
        self.sess = sess
        self.model = model 
        self.data = data
        self.settings = settings
        self.logger = logger 
        self.z_dim = settings['latent_dim']
        self.t = 0 
    
    def train_epoch(self):
        
        loop = tqdm(range(self.settings['num_iter_per_epoch']))
#         loop = range(self.settings['num_iter_per_epoch'])
    
        for ii in loop:
            
            self.t = ii + 1
            
            loss_d, loss_g, loss_auto, gen_sample = self.train_step()
            
            summaries_dict = { 'Decoder': loss_g, 'Discriminator': loss_d, 'AutoEncoder': loss_auto }
            self.logger.summarize(ii, summaries_dict=summaries_dict)
            
            if ii % 10 == 0:
                
                self.plot.plot_sample(ii + 1, 6, gen_sample, self.model.channels)
            
            if ii == 0 :
                
                batch_x, batch_y = next(self.data.next_batch(self.settings["batch_size"]))
                self.plot.plot_sample(ii + 1, 6, batch_x, self.model.channels)
                
            
        self.model.save(self.sess)
        
    def train_step(self):

        G_r = self.settings['G_rounds']
        
        batch_x, batch_y = next(self.data.next_batch(self.settings["batch_size"]))
        batch_z = np.random.uniform(-1, 1, size=(self.settings["batch_size"], self.z_dim, 1, 1))   
#         batch_z = np.float32(np.random.normal(size=[self.settings["batch_size"], self.z_dim, 1, 1]))
        if self.t > np.inf :
            lambda_q = 0 
        else:
            lambda_q = 1.0  / ((self.t)*(self.t))
            
        feed_dict = {self.model.x_input: batch_x, self.model.z_input: batch_z, self.model.t : lambda_q}
        
        d_opt , g_opt, ae_opt, gen_sample = self.sess.run([self.model.d_opt, self.model.g_opt, self.model.ae_opt, self.model.gen_sample], feed_dict= feed_dict)
                
        loss_d = self.model.D_loss.eval(feed_dict = feed_dict, session = self.sess)
        loss_g = self.model.G_loss.eval(feed_dict = feed_dict, session = self.sess)
        loss_auto = self.model.ae_loss.eval(feed_dict = feed_dict, session = self.sess)

        return loss_d, loss_g, loss_auto, gen_sample
    