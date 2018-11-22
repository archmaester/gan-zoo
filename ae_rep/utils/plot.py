import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np 

class Plot(object):
    
    def __init__(self, sess, settings):

        self.sess = sess
        self.settings = settings
        
    
    def plot_sample(self, idx, n_samples, input_p, channels, labels=None):
        
        samples = input_p 
        
        samples = np.array(samples)
        
        assert n_samples <= samples.shape[0]

        if not labels is None:
            assert n_samples <= len(labels)
            if len(labels.shape) > 1 and not labels.shape[1] == 1:
                # one-hot
                label_titles = np.argmax(labels, axis=1)
            else:
                label_titles = labels
        else:
            label_titles = ['NA']*n_samples
        assert n_samples % 2 == 0
        # img_size = int(np.sqrt(samples.shape[1]))

        nrow = int(n_samples/2)
        ncol = 2
        fig, axarr = plt.subplots(nrow, ncol, sharex=True, figsize=(8, 8))
        for m in range(nrow):
            # first column
            sample = samples[m, :, 0]
            # axarr[m, 0].imshow(sample.reshape([img_size,img_size]), cmap='gray')
            axarr[m,0].plot(sample)
            axarr[m, 0].set_title(str(label_titles[m]))
            # second column
            sample = samples[nrow + m, :, 0]
            axarr[m,1].plot(sample)
            # axarr[m, 1].imshow(sample.reshape([img_size,img_size]), cmap='gray')
            axarr[m, 1].set_title(str(label_titles[m + nrow]))
        fig.suptitle(idx)
        fig.subplots_adjust(hspace = 0.15)
        
        fig.savefig(self.settings['dir_root'] +"plots/" + str(idx).zfill(4) + ".png")
        plt.clf()
        plt.close()

        return
        
        
        

