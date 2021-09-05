ffpp = "pattern"

#read csv
import pandas as pd
train = pd.read_csv("../_data/samai/train.csv")
dev = pd.read_csv("../_data/samai/dev.csv")
test = pd.read_csv("../_data/samai/test.csv")

ss = pd.read_csv("../_data/samai/sample_submission.csv")

train = pd.concat([train,dev])
train['ST1_GAP(eV)'] = train['S1_energy(eV)'] - train['T1_energy(eV)']


import kora.install.rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys

import numpy as np
import pandas as pd

import math
train_fps = []#train fingerprints
train_y = [] #train y(label)

for index, row in train.iterrows() : 
  try : 
    mol = Chem.MolFromSmiles(row['SMILES'])
    if ffpp == 'maccs' :    
        fp = MACCSkeys.GenMACCSKeys(mol)
    elif ffpp == 'morgan' : 
        fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 4)
    elif ffpp == 'rdkit' : 
        fp = Chem.RDKFingerprint(mol)
    elif ffpp == 'pattern' : 
        fp = Chem.rdmolops.PatternFingerprint(mol)
    elif ffpp == 'layerd' : 
        fp = Chem.rdmolops.LayeredFingerprint(mol)

    train_fps.append(fp)
    train_y.append(row['ST1_GAP(eV)'])
  except : 
    pass

#fingerfrint object to ndarray
np_train_fps = []
for fp in train_fps:
  arr = np.zeros((0,))
  DataStructs.ConvertToNumpyArray(fp, arr)
  np_train_fps.append(arr)

np_train_fps_array = np.array(np_train_fps)

pd.Series(np_train_fps_array[:,0]).value_counts()

import math
test_fps = []#test fingerprints
test_y = [] #test y(label)

for index, row in test.iterrows() : 
  try : 
    mol = Chem.MolFromSmiles(row['SMILES'])

    if ffpp == 'maccs' :    
        fp = MACCSkeys.GenMACCSKeys(mol)
    elif ffpp == 'morgan' : 
        fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 4)
    elif ffpp == 'rdkit' : 
        fp = Chem.RDKFingerprint(mol)
    elif ffpp == 'pattern' : 
        fp = Chem.rdmolops.PatternFingerprint(mol)
    elif ffpp == 'layerd' : 
        fp = Chem.rdmolops.LayeredFingerprint(mol)

    test_fps.append(fp)
    test_y.append(row['ST1_GAP(eV)'])
  except : 
    pass

np_test_fps = []
for fp in test_fps:
  arr = np.zeros((0,))
  DataStructs.ConvertToNumpyArray(fp, arr)
  np_test_fps.append(arr)

np_test_fps_array = np.array(np_test_fps)

print(np_test_fps_array.shape)
print(len(test_y))

pd.Series(np_test_fps_array[:,0]).value_counts()
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
def create_deep_learning_model():
    model = Sequential()
    model.add(Dense(2048, input_dim=2048, kernel_initializer='random_uniform', activation='softplus'))
    model.add(Dense(1024, activation='softplus'))
    model.add(Dense(100, activation='softplus'))

    model.add(Dense(1, kernel_initializer='random_uniform'))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model

X, Y = np_train_fps_array , np.array(train_y)

#validation
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

estimators = []
# estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=create_deep_learning_model, epochs=10)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=9,random_state=76,shuffle=True)
results = cross_val_score(pipeline, X, Y, cv=kfold)

print("%.2f (%.2f) MAE" % (results.mean(), results.std()))

model = create_deep_learning_model()
model.fit(X, Y, epochs = 50)
test_y = model.predict(np_test_fps_array)
ss['ST1_GAP(eV)'] = test_y

ss.to_csv("pattern_mlp.csv",index=False)