#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from numpy import random
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from utils import ID_COV, FS, COG, VOLS

# Load data
df_original = pd.read_csv('./demo and cognitive data.csv')
vars_keep = ID_COV + FS + COG
data_orginal = df_original[vars_keep]
data_orginal.dropna(inplace=True)

# Remove bad QC
ids_remove = pd.read_csv('hcp_badquality_issuecodes_A_B.csv')['Subject'].tolist()
data = data_orginal[~data_orginal['Subject'].isin(ids_remove)]


def cca(X, pca=None):
    cca_test = list()
    cca_train = list()
    weights = {}

    # Configure bootstrap
    for a in np.arange(0.025, 1.025, 0.025):

        n_iterations = 100
        n_size = int(len(data) * a)

        # Run bootstrap
        corr_n_test = list()
        corr_n_train = list()
        weights_n_fs = list()
        weights_n_cog = list()
        latent_FS_test = list()
        latent_COG_test = list()
        latent_FS_train = list()
        latent_COG_train = list()

        for i in range(n_iterations):
            # Prepare train and test sets
            data_iteration = resample(X, n_samples=n_size, replace=False, random_state=i)
            train, test = train_test_split(data_iteration, test_size=0.20, random_state=i)

            # Normalize data
            scaler = StandardScaler()
            train = scaler.fit_transform(train)
            test = scaler.transform(test)

            # Split modalities
            train_fs = train[:,:len(FS)]
            train_cog = train[:,len(FS):]
            test_fs = test[:,:len(FS)]
            test_cog = test[:,len(FS):]

            # PCA brain
            if pca:
                pca = PCA(n_components=5, random_state=i)
                train_fs = pca.fit_transform(train_fs)
                test_fs = pca.transform(test_fs)
                
            # Fit model
            cca = CCA(n_components=1)
            cca.fit(train_fs, train_cog)

            # Evaluate model
            FS_C_test, COG_C_test = cca.transform(test_fs, test_cog)
            FS_C_train, COG_C_train = cca.transform(train_fs, train_cog)
            C1_corr_test, _ = stats.pearsonr(FS_C_test.flatten(),COG_C_test.flatten())
            C1_corr_train, _ = stats.pearsonr(FS_C_train.flatten(),COG_C_train.flatten())

            # Extract weights
            weights_FS = cca.x_weights_
            weights_COG = cca.y_weights_
            
            # Append correlation and weights
            corr_n_test.append(C1_corr_test)
            corr_n_train.append(C1_corr_train)
            weights_n_fs.append(weights_FS)
            weights_n_cog.append(weights_COG)
            
            # Extract latent variables for N=842
            if n_size == 842:
                latent_FS_test.append(FS_C_test)
                latent_COG_test.append(COG_C_test)
                latent_FS_train.append(FS_C_train)
                latent_COG_train.append(COG_C_train)

        # Append list of correlations for a given sample size
        cca_test.append(corr_n_test)
        cca_train.append(corr_n_train)
        
        # Append weights for a given sample size
        weights_n_fs_df = pd.DataFrame([np.squeeze(i) for i in weights_n_fs])
        weights_n_cog_df = pd.DataFrame([np.squeeze(i) for i in weights_n_cog])
        weights_n = pd.concat([weights_n_fs_df, weights_n_cog_df], ignore_index=True, axis=1)
        cos_sim_n = cosine_similarity(weights_n)
        cos_sim_n2 = cos_sim_n[np.triu_indices(len(cos_sim_n), k = 1)].mean()
        weights[n_size] = cos_sim_n2
        
    return cca_test, cca_train, weights, latent_FS_test, latent_COG_test, latent_FS_train, latent_COG_train


fs = data[FS].values
#tiv = data[['FS_IntraCranial_Vol']].values
#fs = (np.true_divide(fs, tiv)).astype('float32')
cog = data[COG].values
X = np.concatenate([fs,cog], axis=1)

test_pca, train_pca, weights_pca, latent_FS_test, latent_COG_test, latent_FS_train, latent_COG_train = cca(X, pca=True)
#test, train, weights = cca(X, pca=False)
