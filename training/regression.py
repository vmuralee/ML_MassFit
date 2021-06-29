from makedataset import makedataset
import numpy as np
import matplotlib.pyplot as plt

import keras
import sys
import os

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from root_numpy import array2root

data = makedataset('W2LNu10000Events_13Tev.root')
X_train,X_val = data.training_df,data.validation_df
y_train,y_val = data.label_df,data.val_label_df
X_star  = data.testing_df

input_dim = X_train.shape[1]
print('-'*15,' dimensions of the input variables ','-'*15)
print('training   input dimensions',X_train.shape,'  ',y_train.shape)
print('predicitng input dimensions',X_star.shape)
print('-'*50)

fig,ax = plt.subplots(3,3)
ax[0,0].hist(X_train[:,0],50)
ax[0,1].hist(X_train[:,1],50)
ax[0,2].hist(X_train[:,2],50)
ax[1,0].hist(X_train[:,3],50)
ax[1,1].hist(X_train[:,3],50)
ax[1,2].hist(X_train[:,4],50)
ax[2,0].hist(X_train[:,5],50)
ax[2,1].hist(y_train,50)
fig.savefig('dfd.png')
num_pipeline = Pipeline([
    #('imputer',SimpleImputer(strategy="median")),
    ('std_scaler',StandardScaler()),
])

X_train_tr,X_val_tr,X_star_tr = num_pipeline.fit_transform(X_train),num_pipeline.fit_transform(X_val),num_pipeline.fit_transform(X_star)
y_train_tr,y_val_tr = num_pipeline.fit_transform(y_train.reshape(-1,1)),num_pipeline.fit_transform(y_val.reshape(-1,1))
print(y_train_tr)
def plot_loss(history):
    fig1,ax1 = plt.subplots()
    ax1.plot(history.history['loss'], label='loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_ylim([0.5, 1.3])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Error [MPG]')
    ax1.legend()
    fig1.savefig('GANinput_training.png')
    ax1.grid(True)
    
def build_model(layer_geom,learning_rate=3e-3,input_shapes=[8]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shapes))
    for layer in layer_geom:
        model.add(keras.layers.Dense(layer_geom[layer],activation='relu',kernel_initializer="he_normal"))
    he_avg_init = keras.initializers.VarianceScaling(scale=2.,mode='fan_avg',distribution='uniform')
    model.add(keras.layers.Dense(1,activation='tanh',kernel_initializer=he_avg_init))
    optimizer = keras.optimizers.Adam(lr=learning_rate,beta_1=0.5)
    model.compile(loss='mean_squared_error',optimizer=optimizer)
    return model


hlayer_outline = {'hlayer1':32,'hlayer2':64,'hlayer3':128,'hlayer4':64,'hlayer5':32}
model = build_model(hlayer_outline,input_shapes=[input_dim])
    
model.summary()
model_dir = '../data/'
if(sys.argv[1] == 'do_train'):
    fit_model = model.fit(X_train_tr,y_train_tr,epochs=500,batch_size =128,validation_data=(X_val_tr,y_val_tr))#,callbacks=[keras.callbacks.EarlyStopping(patience=10)])
    model.save(model_dir+'GANinput_model.h5')
    plot_loss(fit_model)
elif(sys.argv[1] == 'do_predict'):
    saved_model = keras.models.load_model(model_dir+'regression_model.h5')
    y_pred = saved_model.predict(X_star_tr)
    pred = num_pipeline.inverse_transform(y_pred)
    fig2,ax2 = plt.subplots()
    ax2.hist(pred,50)
    fig2.savefig("fitted_mass.png")
    filename = '../data/W2LNu10000Events_13Tev_new.root',
    #leaf = zip(recopz,recoE)
    # leaf_l = list(zip(y_pred))
    # branch = np.array(leaf_l,dtype=[('reco_mass','float32')])
    # array2root(branch,filename,treename='analysisTree',mode='update')
else:
    print('please put do_train or do_predict')







