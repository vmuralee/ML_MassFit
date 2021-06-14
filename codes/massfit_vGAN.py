################################################################
####    Simple GAN algorithm to construct W reco mass    #######
####                vinaya.krishna@cern.ch               #######
################################################################


from numpy import hstack
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from load_data import load_data
import math
from keras.utils.vis_utils import plot_model

class sGAN_massFit:
    def __init__(self, X_train, X_test,y_true,scale,n_epoch, batch_size):
        self.X_train = X_train
        self.X_test  = X_test
        self.input_dim = X_train.shape[1]
        self.latent_dim  = X_train.shape[1]
        half_batch = int(batch_size/2.0)

        discriminator = self.define_discriminator(self.input_dim)
        generator     = self.define_generator(self.latent_dim,self.input_dim)
        gan_model = self.define_gan(generator,discriminator)
        acc_r,acc_f,ep_ar = [],[],[]
        for i in range(1,n_epoch):
            x_real,y_real = self.generate_real_samples(half_batch)
            #x_input = self.generate_latent_points(X_test,half_batch)
            x_fake,y_fake = self.generate_fake_samples(generator,self.latent_dim,half_batch)
            #print('x real shape ',x_real.shape,' ',y)
            discriminator.train_on_batch(x_real,y_real)
            discriminator.train_on_batch(x_fake,y_fake)
            x_gan = self.generate_latent_points(self.latent_dim,half_batch)
            y_gan = np.ones((half_batch,1))
            gan_model.train_on_batch(x_gan,y_gan)
            #w_reco_mass = self.perform_massfit(self.X_test,generator,scale)
            w_gen_mass = 3200*y_true
            if(i==1 or i%20==0):
                print('='*15,' Epoch ',i,'='*15)
                acc_real,acc_fake = self.perform_massfit(i,generator,discriminator,self.latent_dim,w_gen_mass)
                acc_r.append(acc_real)
                acc_f.append(acc_fake)
                ep_ar.append(i)
        fig,ax = plt.subplots(1,2)
        ax[0].plot(ep_ar,acc_r)
        ax[1].plot(ep_ar,acc_f)
        ax[0].set_ylabel('accuracy on real samples')
        ax[1].set_ylabel('accuracy on fake samples')
        fig.savefig('../plots/training.png')
    def perform_massfit(self, epoch, generator, discriminator, latent_dim,w_gen_mass, n=1000):
        x_real, y_real = self.generate_real_samples(n)
        _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
        x_fake, y_fake = self.generate_fake_samples(generator, latent_dim, n)
        # evaluate discriminator on fake examples
        _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
        print(epoch, acc_real, acc_fake)
        w_reco_mass = 3200*self.fitted_mass(x_fake)
        w_train_mass = 3200*self.fitted_mass(x_real)
        plt.hist(w_reco_mass, 50, density=True, label='W GAN mass')
        plt.hist(w_gen_mass,50,density=True,label='W gen mass')
        plt.hist(w_train_mass,50,density=True,label='W train mass')
        plt.legend()
        plt.savefig(f'../plots/invmass_{epoch}.png')
        plt.close()
        return acc_real,acc_fake

    def fitted_mass(self,reco):
        inv_px = reco[:,4]
        inv_py = reco[:,5]
        inv_pz = reco[:,7]
        inv_e  = np.sqrt(inv_px**2+inv_py**2+inv_pz**2)
    
        muon_px = reco[:,0]
        muon_py = reco[:,1]
        muon_pz = reco[:,2]
        muon_e =  reco[:,3]
        mass2 = (muon_e+inv_e)**2 -((muon_px+inv_px)**2 + (muon_py+inv_py)**2 + (muon_pz+inv_pz)**2)
        invmass = np.sqrt(mass2)#np.array([math.sqrt(i) if i > 0 else math.sqrt(-i) for i in mass2  ])
        return invmass

    def generate_real_samples(self,n):
        ix = np.random.randint(0,self.X_train.shape[0],n)
        x_train = self.X_train[ix]
        y = np.ones((n,1))
        return x_train,y

    def generate_fake_samples(self,generator,latent_dim,n):
        y = np.zeros((n,1))
        x_input = self.generate_latent_points(latent_dim,n)
        X = generator.predict(x_input)
        return X,y

    def generate_latent_points(self,latent_dim,n):
        x_input = np.random.randn(latent_dim*n)
        x_input = x_input.reshape(n, latent_dim)
        return x_input 

    def creating_recoE(self,X, pz_ar):
        px_ar = X[:, 4]
        py_ar = X[:, 5]
        e_ar = np.sqrt(px_ar*px_ar+py_ar*py_ar+pz_ar*pz_ar)
        e_2d_ar = e_ar.reshape(-1, 1)
        X_new1 = np.concatenate((X,pz_ar.reshape(-1,1)), axis=1)
        X_new2 = np.concatenate((X_new1,e_2d_ar),axis=1)
        return X_new2

    # define the standalone discriminator model
    def define_discriminator(self,n_inputs=2):
        model = Sequential()
        model.add(Dense(70, activation='relu',kernel_initializer='he_uniform', input_dim=n_inputs))
        model.add(Dense(35, activation='relu'))
        #model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # compile model
        model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
        plot_model(model,to_file='discriminator_model_summary.png',show_shapes=True,show_layer_names=True)
        return model

    # define the standalone generator model
    def define_generator(self,latent_dim, n_outputs=2):
        model = Sequential()
        model.add(Dense(35, activation='relu',
              kernel_initializer='he_uniform', input_dim=latent_dim))
        model.add(Dense(70, activation='relu'))
        #model.add(Dense(256, activation='relu'))
        model.add(Dense(n_outputs, activation='tanh'))
        plot_model(model,to_file='generator_model_summary.png',show_shapes=True,show_layer_names=True)
        return model

    # define the combined generator and discriminator model, for updating the generator
    def define_gan(self,generator, discriminator):
        # make weights in the discriminator not trainable
        discriminator.trainable = False
        # connect them
        model = Sequential()
        # add generator
        model.add(generator)
        # add the discriminator
        model.add(discriminator)
        # compile model
        model.compile(loss='binary_crossentropy', optimizer='adam')
        plot_model(model,to_file='gan_model_summary.png',show_shapes=True,show_layer_names=True)
        return model
        

X_train,y_train,X_test,y_test,X_input,scale = load_data('W2LNu10000Events_13Tev.root')
fit_result = sGAN_massFit(X_train,X_test,y_test,scale,801,128)
