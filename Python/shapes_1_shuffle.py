
'''
Shuffle the point order for all shapes
'''

import os,unipath
import random
import numpy as np
from matplotlib import pyplot as plt



def random_roll(r):
	n = r.shape[0]  # number of points
	i = random.randint(1, n-1) 
	return np.roll(r, i, axis=0)

def shuffle_points(r):
	n     = r.shape[0]  # number of points
	ind   = np.random.permutation(n)
	return r[ind]


def write_csv(fname, shape, xy):
	with open(fname1, 'w') as f:
		f.write('Shape,X,Y\n')
		for s,(x,y) in zip(shape, xy):
			f.write('%d,%.6f,%.6f\n' %(s,x,y))




# #(0) Shuffle point order for one dataset:
# dirREPO   = unipath.Path( os.path.dirname(__file__) ).parent
# name      = 'Bell'
# fname0    = os.path.join(dirREPO, 'Data', name, 'geom_original.csv')
# a         = np.loadtxt(fname0, delimiter=',', skiprows=1)
# shape     = np.asarray(a[:,0], dtype=int)
# xy        = a[:,1:].copy()
# xy        = np.array( a[:,1:] )
# xy0       = xy.copy()   # copy original ordering for plotting later
# nshapes   = max(shape)
# random.seed(0)
# np.random.seed(0)
# for i in range(nshapes):
# 	b     = shape==(i+1)
# 	xy[b] = random_roll( xy[b] )
# 	xy[b] = shuffle_points( xy[b] )
# ### check ordering for one shape:
# plt.close('all')
# fig,AX  = plt.subplots( 1, 2, figsize=(8,3) )
# ax0,ax1 = AX.flatten()
# r0      = xy0[shape==1]
# r       = xy[shape==1]
# n       = r0.shape[0]
# sc0     = ax0.scatter(r0[:,0], r0[:,1], c=np.arange(n), cmap='jet', vmin=0, vmax=n)
# sc1     = ax1.scatter(r[:,0], r[:,1], c=np.arange(n), cmap='jet', vmin=0, vmax=n)
# cbh = plt.colorbar(sc0, cax=plt.axes([0.91,0.2,0.02,0.6]))
# cbh.set_label('Point number')
# ax0.set_title('Original (ordered points)')
# ax1.set_title('Shuffled points')
# plt.show()




#(1) Shuffle point order for all datasets:
dirREPO   = unipath.Path( os.path.dirname(__file__) ).parent
names     = ['Bell', 'Comma', 'Device8',    'Face', 'Flatfish', 'Hammer',    'Heart', 'Horseshoe', 'Key']
random.seed(0)
np.random.seed(0)
for name in names:
	fname0    = os.path.join(dirREPO, 'Data', name, 'geom_original.csv')
	fname1    = os.path.join(dirREPO, 'Data', name, 'geom_s.csv')
	a         = np.loadtxt(fname0, delimiter=',', skiprows=1)
	shape     = np.asarray(a[:,0], dtype=int)
	xy        = a[:,1:]
	xy0       = xy.copy()   # copy original ordering for plotting later
	nshapes   = max(shape)
	for i in range(nshapes):
		b     = shape==(i+1)
		xy[b] = random_roll( xy[b] )
		xy[b] = shuffle_points( xy[b] )
	write_csv(fname1, shape, xy)






