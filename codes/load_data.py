import ROOT
import numpy as np
from root_numpy import root2array, rec2array, array2root, tree2array
import pandas as pd
from sklearn.model_selection import train_test_split

def read_tree(tree,cols_list):
    tree_arr = tree2array(tree)
    df = pd.DataFrame(tree_arr,columns=cols_list)
    return df

    
def split_data(tree_df):
    traindataset_all,testdataset_all = train_test_split(tree_df,test_size=0.1,random_state=42)
    reco_vars = ['muon_px','muon_py','muon_pz','muon_e','invis_px','invis_py','m_vis']
    gen_vars  = ['genmuon_px','genmuon_py','genmuon_pz','genmuon_e','geninvis_px','geninvis_py','geninvis_e','geninvis_pz','boson_M']
    X_train = traindataset_all.drop(columns=reco_vars+['boson_M']).to_numpy(dtype='float64')
    y_train = traindataset_all['boson_M'].to_numpy(dtype='float64')
    X_test  = testdataset_all.drop(columns=reco_vars+['boson_M','geninvis_pz','geninvis_e']).to_numpy(dtype='float64')
    y_test  = testdataset_all['boson_M'].to_numpy(dtype='float64')
    X_input = tree_df.drop(columns=gen_vars+['m_vis']).to_numpy(dtype='float64')
    scale   = testdataset_all['m_vis'].to_numpy(dtype='float64')
    
    return X_train,y_train,X_test,y_test,X_input,scale

    
def load_data(filename):
    cols_list  = ['genmuon_px','genmuon_py','genmuon_pz','genmuon_e','boson_M']
    cols_list += ['geninvis_px','geninvis_py','geninvis_pz','geninvis_e']
    # cols_list += ['genmet_px','genmet_py','genmet']
    cols_list += ['muon_px','muon_py','muon_pz','muon_e']
    cols_list += ['invis_px','invis_py','m_vis']#,'invis_pz','invis_e']
    # cols_list += ['met_px','met_py','met']
    # cols_list += ['boson_px','boson_py','boson_pz','boson_e','boson_M']

    try:
        file = ROOT.TFile('../data/'+filename)
    except NameError:
        print('file is not found in the directrory')

    treename = 'analysisTree' 
    tree = file.Get(treename)

    tree_df = read_tree(tree,cols_list)

    #print(tree_df.head(5))

    X_train,y_train,X_test,y_test,X_input,scale = split_data(tree_df)

    print('='*15,' Dataset Info ','='*15)
    print('Training input shape: ',X_train.shape,' muon_px, muon_py, muon_pz, muon_e, invis_px, invis_py, invis_e, invis_pz')
    print('Test  input shape: ',X_test.shape,' muon_px, muon_py, muon_pz, muon_e, invis_px, invis_py')
    print('='*15,' Dataset Info ','='*15)

    return X_train,y_train,X_test,y_test,X_input,scale




## Testing script ##
# X_train,X_valid,X_input,y_true = load_data('W2LNu10000Events_13Tev.root')
# print('training shape: ',X_train.shape,' test shape: ',X_valid.shape,' input shape: ',X_input.shape)
# noise = np.random.normal(0,1,(X_valid.shape[0],))
# print('shape of noise vector',noise.shape)
# noise_var = np.concatenate((X_valid,noise.reshape(1,-1).T),axis=1)
# print('shape of input vector',noise_var.shape)
# print('latent dimension: ',noise_var.shape[1])
# reco_momenta = np.concatenate((X_valid,(noise_var[:,-1].reshape(1,-1)).T),axis=1)
# print('readdited input dimension: ',reco_momenta.shape)
# print('true value shape: ',y_true)