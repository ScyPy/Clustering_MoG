import numpy as np
import os
import matplotlib.pyplot as plt
from print_values import *
from plot_data_all_phonemes import *
from plot_data import *
import random
from sklearn.preprocessing import normalize
from get_predictions import *
from plot_gaussians import *

# File that contains the data
data_npy_file = 'data/PB_data.npy'

# Loading data from .npy file
data = np.load(data_npy_file, allow_pickle=True)
data = np.ndarray.tolist(data)

# Make a folder to save the figures
figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# Array that contains the phoneme ID (1-10) of each sample
phoneme_id = data['phoneme_id']
# frequencies f1 and f2
f1 = data['f1']
f2 = data['f2']

# Initialize array containing f1 & f2, of all phonemes.
X_full = np.zeros((len(f1), 2))
#########################################
# Write your code here
# Store f1 in the first column of X_full, and f2 in the second column of X_full
for i in range(len(X_full)):
    X_full[i].put(indices=0, values=f1[i])
    X_full[i].put(indices=1, values=f2[i])
########################################/
X_full = X_full.astype(np.float32)

# number of GMM components
k = 3
p_id_1 = 1
p_id_2 = 2
#########################################
# Write your code here

# Create an array named "X_phonemes_1_2", containing only samples that belong to phoneme 1 and samples that belong to phoneme 2.
# The shape of X_phonemes_1_2 will be two-dimensional. Each row will represent a sample of the dataset, and each column will represent a feature (e.g. f1 or f2)
# Fill X_phonemes_1_2 with the samples of X_full that belong to the chosen phonemes
# To fill X_phonemes_1_2, you can leverage the phoneme_id array, that contains the ID of each sample of X_full

# X_phonemes_1_2 = ...
X_phoneme = X_full[phoneme_id == p_id_1, :]

X_phonemes_1_2 = X_full[np.logical_or(phoneme_id==1, phoneme_id==2), :]
#And
########################################/

# Plot array containing the chosen phonemes

# Create a figure and a subplot
fig, ax1 = plt.subplots()

title_string = 'Phoneme 1 & 2'
# plot the samples of the dataset, belonging to the chosen phoneme (f1 & f2, phoneme 1 & 2)
plot_data(X=X_phonemes_1_2, title_string=title_string, ax=ax1)
# save the plotted points of phoneme 1 as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'dataset_phonemes_1_2.png')
plt.savefig(plot_filename)


#########################################
# Write your code here
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 1
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 2
# Compare these predictions for each sample of the dataset, and calculate the accuracy, and store it in a scalar variable named "accuracy"

########################################


#Declare models
npy_filename_1 = 'data/GMM_params_phoneme_{:02}_k_{:02}.npy'.format(p_id_1, k)
npy_filename_2 = 'data/GMM_params_phoneme_{:02}_k_{:02}.npy'.format(p_id_2, k)
#Load in phoneme1
npy_1 = np.load(npy_filename_1, allow_pickle=True)

#Convert to list to extract parameters
npy_list_1 = np.ndarray.tolist(npy_1)

#Extract mu, s p
mu = npy_list_1['mu']
s = npy_list_1['s']
p = npy_list_1['p']

#Copy phonemes1_2 dataset
X_phonemes_1_2_copy = X_phonemes_1_2.copy()

#Run get_predictions GMM, output P(X|mu,sigma) (high dimensionality)
pred_1 = get_predictions(mu,s,p,X_phonemes_1_2_copy)


#Sum
phoneme_1_pred = pred_1.sum(axis=1)
# for i in pred_1:
#     phenome_1_pred.append(i[0] + i[1] + i[2])

#Phoneme 2
#Load in phoneme 2
npy_2 = np.load(npy_filename_2,allow_pickle=True)

#convert to list
npy_list_2 = np.ndarray.tolist(npy_2)

mu_2 = npy_list_2['mu']
s_2 = npy_list_2['s']
p_2 = npy_list_2['p']


##Get Predictions for phoneme 2
pred_2 = get_predictions(mu_2,s_2,p_2,X_phonemes_1_2_copy)
#Sum
phoneme_2_pred = pred_2.sum(axis=1)
# for i in pred_2:
#     phenome_2_pred.append(i[0] + i[1] + i[2])


#Predictions
all_preds = np.ones(len(X_phonemes_1_2_copy)) * 2

all_preds[phoneme_1_pred > phoneme_2_pred] = 1 ##Change




headers = phoneme_id[np.logical_or(phoneme_id==1, phoneme_id==2)]
accuracy = (1 - (np.sum(all_preds != headers) / X_phonemes_1_2_copy.shape[0]))
print('Accuracy using GMMs with {} components: {:.2f}%'.format(k, accuracy * 100))

################################################
# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()