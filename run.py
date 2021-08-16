import os
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as pl
from bhmtorch_cpu import BHM2D_PYTORCH
from bhm_utils import getPartitions

def load_parameters(case):
    parameters = \
        {'kitti1': \
             ( os.path.abspath('../../datasets/kitti/kitti2011_09_26_drive0001_frame'),
              (2, 2), #hinge point resolution
              (-80, 80, -80, 80), #area [min1, max1, min2, max2]
              None,
              None,
              0.5, #gamma
              ),

        'intel': \
             ('datasets/intel.csv',
              (0.5, 0.5), #x1 and x2 resolutions for positioning hinge points
              (-20, 20, -25, 10), #area to be mapped [x1_min, x1_max, x2_min, x2_max]
              1, #N/A
              0.01, #threshold for filtering data
              6.71 #gamma: kernel parameter
            ),

         }

    return parameters[case]

# Settings
dtype = torch.float32
device = torch.device("cpu")
dataset =  'intel'
#device = torch.device("cuda:0") # Uncomment this to run on GPU

# Read the file
fn_train, cell_resolution, cell_max_min, skip, thresh, gamma = load_parameters(dataset)

#read data
dataset = pd.read_csv(fn_train, delimiter=',').values
print(f"Dataset Shape: {dataset.shape}")
dataset = torch.tensor(dataset, dtype=torch.float32)
X_train = dataset[:, 0:3]
Y_train = dataset[:, 3].reshape(-1, 1)

max_t = len(torch.unique(X_train[:, 0]))
print(f"Total Timesteps: {max_t}")




# =================================== Core ========================================
for ith_scan in range(0, max_t, skip):

    # extract data points of the ith scan
    ith_scan_indx = X_train[:, 0] == ith_scan
    print('{}th scan:\n  N={}'.format(ith_scan, torch.sum(ith_scan_indx)))
    X_new = X_train[ith_scan_indx, 1:]
    y_new = Y_train[ith_scan_indx]

    if ith_scan == 0:
        # get all data for the first scan and initialize the model
        X, y = X_new, y_new
        bhm_mdl = BHM2D_PYTORCH(
        	gamma=gamma,
        	grid=None,
        	cell_resolution=cell_resolution,
        	cell_max_min=cell_max_min,
        	X=X,
        	nIter=1,
        )
    else:
        # information filtering
        # what is the purpose of this? Check ipynb
        q_new = bhm_mdl.predict(X_new).reshape(-1, 1)
        info_val_indx = torch.absolute(q_new - y_new) > thresh
        info_val_indx = info_val_indx.flatten()
        X, y = X_new[info_val_indx, :], y_new[info_val_indx]
        print('  {:.2f}% points were used.'.format(X.shape[0]/X_new.shape[0]*100))

    # Fit the model
    t1 = time.time()
    bhm_mdl.fit(X, y)
    t2 = time.time()

    # query the model
    q_resolution = 0.25
    xx, yy= np.meshgrid(np.arange(cell_max_min[0], cell_max_min[1] - 1, q_resolution),
                         np.arange(cell_max_min[2], cell_max_min[3] - 1, q_resolution))
    grid = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis]))
    Xq = torch.tensor(grid, dtype=torch.float32)
    yq = bhm_mdl.predict(Xq)

    Xq = Xq.cpu().numpy()
    yq = yq.cpu().numpy()

    # ======================== Plotting at ith timestep ============================
    print('Plotting...')
    pl.figure(figsize=(13,5))
    pl.subplot(121)
    ones_ = np.where(y==1)
    pl.scatter(X[ones_, 0], X[ones_, 1], c='r', cmap='jet', s=5)
    pl.title('Laser hit points at t={}'.format(ith_scan))
    pl.xlim([cell_max_min[0], cell_max_min[1]]); pl.ylim([cell_max_min[2], cell_max_min[3]])
    pl.subplot(122)
    pl.title('SBHM at t={}'.format(ith_scan))
    pl.scatter(Xq[:, 0], Xq[:, 1], c=yq, cmap='jet', s=10, marker='8',)
    pl.colorbar()
    pl.xlim([cell_max_min[0], cell_max_min[1]]); pl.ylim([cell_max_min[2], cell_max_min[3]])
    pl.savefig(os.path.abspath('outputs/intel_{:03d}.png'.format(ith_scan)), bbox_inches='tight')

