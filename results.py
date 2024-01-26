# This file is for generating plots of the data

import os, fnmatch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# File locations
f_root = "/home/user1/Documents/ML_DWT/"
f_model = "Models/"
f_output = "Final Test/"
fileloc_model = f"{f_root}{f_model}"
fileloc_output = f"{f_root}{f_output}"

# Establish model tags
model_tags = ['MLP_BCV', 'LSTM_BCV', 'GRU_BCV', 'MLP_HO', 'LSTM_HO', 'GRU_HO']
# Create empty array where all data will be populated
# There are 6 different model tags and 9 sizes of each model type
algos = np.empty((6, 3, 9), dtype=object)
# Iterate through all the model tags
for i, tag in enumerate(model_tags):
    # Load a list of all the models matching the tag
    file_list = sorted(fnmatch.filter(os.listdir(fileloc_model), f'{tag}*.txt'))
    # Iterate through all the models and fill column 0 with their sizes
    for j, file in enumerate(file_list):
        algos[i, 0, j] = file[file.find('hidden-')+len('hidden-'):file.rfind('_')]
        # Fill column 1 with training times
        with open(file, 'r') as f:
            file_text = f.read()
            f.close()
            algos[i, 1, j] = float(file_text[file_text.find('Training time = ')+len('Training time ='): \
                                       file_text.find(' minutes')])
        # Fill column 2 with reported test RMSE
        algos[i, 2, j] = file[file.find('.')-1:file.rfind('.')]            
    

# Load the file that contains the production set RMSE and MAE of each model
data = np.loadtxt(fileloc_output + 'RMSE.csv', delimiter=',', dtype=object)
mae_t, mae_p = data[:, 2].reshape((6, 9)).astype(np.float), data[:, 3].reshape((6, 9)).astype(np.float)

# Add production set data to the array
mae_t = np.expand_dims(mae_t, axis=1)
mae_p = np.expand_dims(mae_p, axis=1)
algos = np.append(algos, mae_t, 1)
algos = np.append(algos, mae_p, 1)

#%%
# Plot test set performance
fig = plt.figure(figsize=(5.3, 5.3))
plt.boxplot(algos[:3, 2, :].T,
            labels=['MLP', 'LSTM', 'GRU'],
            boxprops=dict(lw=3),
            capprops=dict(lw=3),
            whiskerprops=dict(lw=3),
            flierprops=dict(ms=7, mew=3),
            medianprops=dict(lw=3, color='red'))
# to show missing flier point
plt.plot(1, 0.0985, 'k^', ms=7, mew=3)
plt.text(1.05, 0.095, '0.157', fontsize=14, weight='bold')
plt.xlabel('Model type', fontsize=16, weight='bold')
plt.ylabel('Test set MAE (NTU)', fontsize=16, weight='bold')
plt.xticks(fontsize=16, weight='bold')
plt.yticks([0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10], fontsize=16, weight='bold')
plt.ylim((0.03,0.1))
plt.title('Models trained using BCV', fontsize=16, weight='bold')

fig = plt.figure(figsize=(5.3, 5.3))
plt.boxplot(algos[3:, 2, :].T,
            labels=['MLP', 'LSTM', 'GRU'],
            boxprops=dict(lw=3),
            capprops=dict(lw=3),
            whiskerprops=dict(lw=3),
            flierprops=dict(ms=7, mew=3),
            medianprops=dict(lw=3, color='red'))
# to show missing flier point
plt.plot(1, 0.0985, 'k^', ms=7, mew=3)
plt.text(1.05, 0.095, '0.187', fontsize=14, weight='bold')
plt.xlabel('Model type', fontsize=16, weight='bold')
plt.ylabel('Test set MAE (NTU)', fontsize=16, weight='bold')
plt.xticks(fontsize=16, weight='bold')
plt.yticks([0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10], fontsize=16, weight='bold')
plt.ylim((0.03,0.1))
plt.title('Models trained using holdout', fontsize=16, weight='bold')

#%% Plot training times
fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')
cmap = mpl.colormaps['RdYlGn_r']
z_min, z_max = 0, np.max(algos[:, 1, :])/60

# MLP
x, y = 0.5 + np.arange(3), 1.5 + np.arange(3)
xx, yy = np.meshgrid(x, y)
xr, yr = xx.ravel(), yy.ravel()
zr = algos[0, 1, :]/60
dx, dy, dz = 1, 1, np.zeros(zr.shape)
clr = [cmap((k-z_min)/z_max) for k in zr]
ax1.bar3d(xr, yr, dz, dx, dy, zr, shade=True, color=clr)
ax1.azim = -45
ax1.set_xticks([1, 2, 3], ['960   ', '1920    ', '3840    '], rotation=45, rotation_mode='anchor', va='baseline')
ax1.set_yticks([2, 3, 4], [' 2', ' 3', ' 4'], rotation=-45, rotation_mode='anchor', va='baseline')
ax1.set_zticks([0, 8, 16, 24, 32])
ax1.invert_xaxis()
ax1.set_title('MLP', y=1, fontsize=16, weight='bold')
ax1.xaxis.set_tick_params(labelsize=14)
ax1.yaxis.set_tick_params(labelsize=14)
ax1.zaxis.set_tick_params(labelsize=14)
ax1.set_box_aspect(aspect=None, zoom=1)
ax1.set_xlabel('\n\nHidden size', fontsize=14)
# ax1.set_ylabel('\nHidden layers', fontsize=14)

# LSTM
zr = algos[1, 1, :]/60
clr = [cmap((k-z_min)/z_max) for k in zr]
ax2.bar3d(xr, yr, dz, dx, dy, zr, shade=True, color=clr)
ax2.azim = -45
ax2.set_xticks([1, 2, 3], ['32  ', '64  ', '128   '], rotation=45, rotation_mode='anchor', va='baseline')
ax2.set_yticks([2, 3, 4], [' 2', ' 3', ' 4'], rotation=-45, rotation_mode='anchor', va='baseline')
ax2.set_zticks([0, 8, 16, 24, 32])
ax2.invert_xaxis()
ax2.set_title('LSTM', y=1, fontsize=16, weight='bold')
ax2.xaxis.set_tick_params(labelsize=14)
ax2.yaxis.set_tick_params(labelsize=14)
ax2.zaxis.set_tick_params(labelsize=14)
ax2.set_box_aspect(aspect=None, zoom=1)
# ax2.set_xlabel('\n\nHidden size', fontsize=14)
# ax2.set_ylabel('\nHidden layers', fontsize=14)

# GRU
zr = algos[2, 1, :]/60
clr = [cmap((k-z_min)/z_max) for k in zr]
ax3.bar3d(xr, yr, dz, dx, dy, zr, shade=True, color=clr)
ax3.azim = -45
ax3.set_xticks([1, 2, 3], ['32  ', '64  ', '128   '], rotation=45, rotation_mode='anchor', va='baseline')
ax3.set_yticks([2, 3, 4], [' 2', ' 3', ' 4'], rotation=-45, rotation_mode='anchor', va='baseline')
ax3.set_zticks([0, 8, 16, 24, 32])
ax3.invert_xaxis()
ax3.set_title('GRU', y=1, fontsize=16, weight='bold')
ax3.xaxis.set_tick_params(labelsize=14)
ax3.yaxis.set_tick_params(labelsize=14)
ax3.zaxis.set_tick_params(labelsize=14)
ax3.set_box_aspect(aspect=None, zoom=1)
# ax3.set_xlabel('\n\nHidden size', fontsize=14)
ax3.set_ylabel('\nHidden layers', fontsize=14)
ax3.set_zlabel('\nTraining time (h)', fontsize=14)

#%% Plot production set performance
fig = plt.figure(figsize=(12, 5))

plt.boxplot(algos[:, 3, :].T,
            labels=['MLP\n(BCV)', 'LSTM\n(BCV)', 'GRU\n(BCV)', 'MLP\n(h/o)', 'LSTM\n(h/o)', 'GRU\n(h/o)'],
            boxprops=dict(lw=3),
            capprops=dict(lw=3),
            whiskerprops=dict(lw=3),
            flierprops=dict(ms=7, mew=3),
            medianprops=dict(lw=3, color='red'))
# to show missing flier point
plt.plot(4, 0.108, 'k^', ms=7, mew=3)
plt.text(4.05, 0.104, '0.185', fontsize=14, weight='bold')
plt.xlabel('Model type\n(training method)', fontsize=16, weight='bold')
plt.ylabel('Production set MAE (NTU)', fontsize=16, weight='bold')
plt.xticks(fontsize=16, weight='bold')
plt.yticks([0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11], fontsize=16, weight='bold')
plt.ylim((0.04, 0.11))

#%% Plot test vs production performance
fig = plt.figure(figsize=(5, 5))
bcv_x = np.append(algos[1, 2, :].reshape(-1), algos[2, 2, :].reshape(-1))
bcv_y = np.append(algos[1, 3, :].reshape(-1), algos[2, 3, :].reshape(-1))
ho_x = np.append(algos[4, 2, :].reshape(-1), algos[5, 2, :].reshape(-1))
ho_y = np.append(algos[4, 3, :].reshape(-1), algos[5, 3, :].reshape(-1))

fig = plt.figure(figsize=(5, 5))
plt.plot(algos[4, 2, :].reshape(-1), algos[4, 3, :].reshape(-1), 'ko', mfc='orange', ms=8)
plt.plot(algos[5, 2, :].reshape(-1), algos[5, 3, :].reshape(-1), 'ko', mfc='green', ms=8)
plt.plot([0.03, 0.07], [0.03, 0.07], 'k-')
plt.xlabel('Test set MAE (NTU)', fontsize=16, weight='bold')
plt.ylabel('Production set MAE (NTU)', fontsize=16, weight='bold')
plt.xticks([0.03, 0.04, 0.05, 0.06, 0.07], fontsize=16, weight='bold')
plt.yticks([0.03, 0.04, 0.05, 0.06, 0.07], fontsize=16, weight='bold')
plt.legend(['LSTM', 'GRU'])

fig = plt.figure(figsize=(5, 5))
plt.plot(algos[3, 2, :].reshape(-1), algos[3, 3, :].reshape(-1), 'ko', mfc='magenta', ms=8)
plt.plot([0.03, 0.19], [0.03, 0.19], 'k-')
plt.xlabel('Test set MAE (NTU)', fontsize=16, weight='bold')
plt.ylabel('Production set MAE (NTU)', fontsize=16, weight='bold')
plt.xticks([0.03, 0.07, 0.11, 0.15, 0.19], fontsize=16, weight='bold')
plt.yticks([0.03, 0.07, 0.11, 0.15, 0.19], fontsize=16, weight='bold')
plt.legend(['MLP'])