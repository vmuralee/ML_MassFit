import ROOT
import numpy as np
from numpy import testing
from root_numpy import root2array, rec2array, array2root, tree2array
import pandas as pd
from sklearn.model_selection import train_test_split

class makedataset:
    def __init__(self,filename):
        try:
            file = ROOT.TFile('../data/'+filename)
        except NameError:
            print('file is not found in the directrory')

        treename = 'analysisTree' 
        tree = file.Get(treename)
        cols_list = self.getListofBranch()
        tree_df = self.read_tree(tree,cols_list)

        self.recoFeatures = tree_df.drop(columns=['genmuon_px','genmuon_py','genmuon_pz','genmuon_e','geninvis_px','geninvis_py','geninvis_pz','geninvis_e','boson_M']).to_numpy(dtype='float32')
        self.genLabel    = tree_df['boson_M'].to_numpy(dtype='float32')
        self.genFeatures  = tree_df.drop(columns=['muon_px','muon_py','muon_pz','muon_e','invis_px','invis_py','boson_M']).to_numpy(dtype='float32')
        
    def update_rootfile(self,newfile,treename,pz_array):
        filename = '../data/'+newfile
        branch = np.array(pz_array,dtype=[('inv_pz','float32')])
        return array2root(branch,filename,treename,mode='update')

    def getGenFeatureVals(self):
        return self.genFeatures

    def getGenLabelVals(self):
        return self.genLabel

    def getRecoFeatureVals(self):
        return self.recoFeatures
    
    def read_tree(self,tree,cols_list):
        tree_arr = tree2array(tree)
        df = pd.DataFrame(tree_arr,columns=cols_list)
        return df

    def getListofBranch(self):
        cols_list = ['geninvis_px','geninvis_py','geninvis_pz','geninvis_e']
        cols_list += ['genmuon_px','genmuon_py','genmuon_pz','genmuon_e']
        cols_list += ['muon_px','muon_py','muon_pz','muon_e']
        cols_list += ['invis_px','invis_py']
        cols_list += ['boson_M']
        print(cols_list)
        return cols_list
