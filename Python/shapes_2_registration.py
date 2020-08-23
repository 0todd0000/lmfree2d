
'''
Register contour points using the coherent point drift (CPD) algorithm
'''

import os,unipath
import numpy as np
from matplotlib import pyplot as plt
import pycpd



def register_cpd_single_pair(r0, r1):
	reg     = pycpd.RigidRegistration(X=r0, Y=r1)	
	reg.register()
	r1r     = reg.TY
	return r1r


def register_cpd_dataset(xy, shape):
	### get template shape:
	u         = np.unique(shape)
	npoints   = [(shape==uu).sum() for uu in u]  # number of points for each shape
	ind       = np.argmax(npoints)  # choose the shape with
	r0        = xy[shape==u[ind]]
	### register:
	xyr       = xy.copy()
	for uu in u:
		if uu!= u[ind]:
			i      = shape==uu
			xyr[i] = register_cpd_single_pair(r0, xy[i])
	return xyr,r0


def stack(xy, shape):
	return np.array( [xy[shape==u]  for u in np.unique(shape)] )

def write_csv(fname, shape, xy):
	with open(fname1, 'w') as f:
		f.write('Shape,X,Y\n')
		for s,(x,y) in zip(shape, xy):
			f.write('%d,%.6f,%.6f\n' %(s,x,y))




# #(0) Register one shape pair (single CPD iteration):
# dirREPO   = unipath.Path( os.path.dirname(__file__) ).parent
# name      = 'Bell'
# fname0    = os.path.join(dirREPO, 'Data', name, 'geom_s.csv')
# a         = np.loadtxt(fname0, delimiter=',', skiprows=1)
# shape     = np.asarray(a[:,0], dtype=int)
# xy        = a[:,1:]
# r0        = xy[shape==1]  # template shape
# r1        = xy[shape==2]  # source shape
# r1r       = register_cpd_single_pair(r0, r1)
# ### check registration:
# plt.close('all')
# fig,AX  = plt.subplots( 1, 2, figsize=(8,3) )
# ax0,ax1 = AX.flatten()
# sc0     = ax0.scatter(r0[:,0], r0[:,1], color='b')
# sc1     = ax0.scatter(r1[:,0], r1[:,1], color='r')
# ax0.axis('equal')
# sc0     = ax1.scatter(r0[:,0], r0[:,1], color='b')
# sc1     = ax1.scatter(r1r[:,0], r1r[:,1], color='r')
# ax1.axis('equal')
# plt.show()



# #(1) Register one dataset:
# dirREPO   = unipath.Path( os.path.dirname(__file__) ).parent
# name      = 'Bell'
# fname0    = os.path.join(dirREPO, 'Data', name, 'geom_s.csv')
# a         = np.loadtxt(fname0, delimiter=',', skiprows=1)
# shape     = np.asarray(a[:,0], dtype=int)
# xy        = a[:,1:]
# xyr,temp  = register_cpd_dataset(xy, shape)
# ### check registration:
# plt.close('all')
# fig,AX    = plt.subplots( 1, 2, figsize=(8,3) )
# ax0,ax1   = AX.flatten()
# r0,r      = stack(xy, shape), stack(xyr, shape)
# for rr in r0:
# 	ax0.scatter(rr[:,0], rr[:,1], color='0.7', s=3)
# for rr in r:
# 	ax1.scatter(rr[:,0], rr[:,1], color='0.7', s=3)
# ax0.scatter(temp[:,0], temp[:,1], color='b', s=5)
# ax1.scatter(temp[:,0], temp[:,1], color='b', s=5)
# ax0.axis('equal')
# ax1.axis('equal')
# ax0.set_title('Original', size=14)
# ax1.set_title('Registered', size=14)
# plt.show()



#(2) Register and save all datasets:
dirREPO   = unipath.Path( os.path.dirname(__file__) ).parent
names     = ['Bell', 'Comma', 'Device8',    'Face', 'Flatfish', 'Hammer',    'Heart', 'Horseshoe', 'Key']
for name in names:
	print( f'Registering {name} dataset...' )
	fname0    = os.path.join(dirREPO, 'Data', name, 'geom_s.csv')
	fname1    = os.path.join(dirREPO, 'Data', name, 'geom_sr.csv')
	a         = np.loadtxt(fname0, delimiter=',', skiprows=1)
	shape     = np.asarray(a[:,0], dtype=int)
	xy        = a[:,1:]
	xyr,_     = register_cpd_dataset(xy, shape)
	write_csv(fname1, shape, xyr)
