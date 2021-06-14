################################################################
####    Conditional GAN algorithm to construct W reco mass  ####
####                vinaya.krishna@cern.ch                  ####
################################################################

import numpy as np
from numpy import hstack
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn

from keras.models import Sequential
from keras.layers import Dense
from keras.models import Model
from keras.layers import Input
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Concatenate
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

from matplotlib import pyplot
import matplotlib.pyplot as plt
from load_data import load_data
import math
import os


class cGAN_massFit:
    def __init__(self, X_train,y_label, X_test,y_true,scale,n_epoch, batch_size):
        self.X_train = X_train
        self.X_test  = X_test
        self.y_label = y_label
        self.input_dim = X_train.shape[1]
        self.latent_dim  = X_train.shape[1]
        half_batch = int(batch_size/2.0)

        discriminator = self.define_discriminator(self.input_dim)
        generator     = self.define_generator(self.latent_dim,self.input_dim)
        gan_model = self.define_gan(generator,discriminator)
        try:
            path_to_store_graphs = '/mnt/e/ML_massfit/plots/cGAN_plots' 
            os.mkdir(path_to_store_graphs)
        except:
            print('Image folder already created ...')
        acc_r,acc_f,ep_ar = [],[],[]
        for i in range(1,n_epoch):
            [x_real,label_real],y_real = self.generate_real_samples(half_batch)
            #x_input = self.generate_latent_points(X_test,half_batch)
            [x_fake,label_fake],y_fake = self.generate_fake_samples(generator,self.latent_dim,half_batch)
            #print('x real shape ',x_real.shape,' ',y)
            discriminator.train_on_batch([x_real,label_real],y_real)
            discriminator.train_on_batch([x_fake,label_fake],y_fake)
            [x_gan,label_gan] = self.generate_latent_points(self.latent_dim,half_batch)
            y_gan = np.ones((half_batch,1))
            gan_model.train_on_batch([x_gan,label_gan],y_gan)
            #w_reco_mass = self.perform_massfit(self.X_test,generator,scale)
            w_gen_mass = 3200*y_true
            if(i==1 or i%20==0):
                print('='*15,' Epoch ',i,'='*15)
                self.perform_massfit(i,generator,discriminator,self.latent_dim,w_gen_mass)
                
      
    def perform_massfit(self, epoch, generator, discriminator, latent_dim,w_gen_mass, n=1000):
        [x_real,label_real], y_real = self.generate_real_samples(n)
        #_, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
        [x_fake,label_fake], y_fake = self.generate_fake_samples(generator, latent_dim, n)
        # evaluate discriminator on fake examples
        #_, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
        #print(epoch, acc_real, acc_fake)
        w_reco_mass = 3200*self.fitted_mass(x_fake)
        w_train_mass = 3200*self.fitted_mass(x_real)
        plt.hist(w_reco_mass, 50, density=True, label='W GAN mass')
        plt.hist(w_gen_mass,50,density=True,label='W gen mass')
        plt.hist(w_train_mass,50,density=True,label='W train mass')
        plt.legend()
        plt.savefig(f'/mnt/e/ML_massfit/plots/cGAN_plots/invmass_{epoch}.png')
        plt.close()

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
        x_train,y_train = self.X_train[ix],self.y_label[ix]
        y = np.ones((n,1))
        return [x_train,y_train],y

    def generate_fake_samples(self,generator,latent_dim,n):
        y = np.zeros((n,1))
        x_input,y_input = self.generate_latent_points(latent_dim,n)
        X = generator.predict([x_input,y_input])
        return [X,y_input],y

    def generate_latent_points(self,latent_dim,n):
        x_input = np.random.randn(latent_dim*n)
        y_input = np.random.normal(80,0.5,n)/3200
        x_input = x_input.reshape(n, latent_dim)
        return [x_input,y_input] 

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
        in_label = Input(shape=(1,))
        li = Embedding(1,5)(in_label)
        li =Dense(n_inputs)(li)
        li = Reshape((n_inputs,))(li)
        in_layer = Input(shape=(n_inputs,))
        merge = Concatenate()([in_layer,li])
        fe = Dense(70,activation='relu',kernel_initializer='he_uniform')(merge)
        fe = LeakyReLU(alpha=0.1)(fe)
        fe = Dense(35,activation='relu')(fe)
        fe = Dropout(0.4)(fe)
        out_layer = Dense(1, activation='sigmoid')(fe)
        model = Model([in_layer, in_label], out_layer)
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy',
                  optimizer=opt, metrics=['accuracy'])
        plot_model(model,to_file='/mnt/e/ML_massfit/plots/cGAN_plots/cDiscri_model_summary.png',show_shapes=True,show_layer_names=True)
        return model

    # define the standalone generator model
    def define_generator(self,latent_dim, n_outputs=2):
        in_label = Input(shape=(1,))
        li = Embedding(1,5)(in_label)
        li =Dense(n_outputs)(li)
        li = Reshape((n_outputs,))(li)
        in_lat = Input(shape=latent_dim)
        gen = Dense(n_outputs)(in_lat)
        gen = LeakyReLU(alpha=0.1)(gen)
        merge = Concatenate()([gen, li])
        gen = Dense(70,activation='relu')(merge)
        gen = Dense(35,activation='relu')(gen)
        gen = Dropout(0.4)(gen)
        out_layer = Dense(latent_dim,activation='tanh')(gen)
        model = Model([in_lat, in_label], out_layer)
        plot_model(model,to_file='/mnt/e/ML_massfit/plots/cGAN_plots/cGenera_model_summary.png',show_shapes=True,show_layer_names=True)
        return model

    # define the combined generator and discriminator model, for updating the generator
    def define_gan(self,generator, discriminator):
        # make weights in the discriminator not trainable
        discriminator.trainable = False
        gen_noise,gen_label =  generator.input
        gen_output = generator.output
        gan_output = discriminator([gen_output,gen_label])
        model = Model([gen_noise, gen_label], gan_output)
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        plot_model(model,to_file='/mnt/e/ML_massfit/plots/cGAN_plots/cGAN_model_summary.png',show_shapes=True,show_layer_names=True)
        return model
        

X_train,y_train,X_test,y_test,X_input,scale = load_data('W2LNu10000Events_13Tev.root')
fit_result = cGAN_massFit(X_train,y_train,X_test,y_test,scale,801,128)
