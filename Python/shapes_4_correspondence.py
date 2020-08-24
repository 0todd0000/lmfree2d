
'''
Find optimum correspondence between shapes
'''

import os,unipath
import numpy as np
from matplotlib import pyplot as plt
from geomdl import fitting





def corresp_roll(r0, r1):
	f       = []
	n       = r1.shape[0]
	for i in range(n):
		r   = np.roll(r1, i, axis=0)
		f.append( sse(r0, r) )
	i       = np.argmin(f)
	return np.roll(r1, i, axis=0)


def set_npoints(r, n):
	pcurve    = fitting.interpolate_curve(list(r), 3)
	pcurve.sample_size = n
	return np.asarray( pcurve.evalpts )


def sse(r0, r1):
	return (np.linalg.norm(r1-r0, axis=1)**2).sum()


def write_csv(fname, r):
	shape  = np.hstack([[i+1]*rr.shape[0]  for i,rr in enumerate(r)])
	with open(fname1, 'w') as f:
		f.write('Shape,X,Y\n')
		for s,(x,y) in zip(shape, np.vstack(r)):
			f.write('%d,%.6f,%.6f\n' %(s,x,y))




# #(0) Find optimum-order correspondence between two shapes:
# dirREPO   = unipath.Path( os.path.dirname(__file__) ).parent
# name      = 'Bell'
# fname0    = os.path.join(dirREPO, 'Data', name, 'geom_sro.csv')
# a         = np.loadtxt(fname0, delimiter=',', skiprows=1)
# shape     = np.asarray(a[:,0], dtype=int)
# xy        = a[:,1:]
# r0        = xy[shape==1]
# r1        = xy[shape==2]
# n         = max(r0.shape[0], r1.shape[0])
# r0        = set_npoints(r0, n)
# r1        = set_npoints(r1, n)
# r2        = corresp_roll(r0, r1)
# ### check ordering:
# plt.close('all')
# fig,AX  = plt.subplots( 1, 3, figsize=(12,3) )
# ax0,ax1,ax2 = AX.flatten()
# ax0.scatter(r0[:,0], r0[:,1], c=np.arange(n), cmap='jet', vmin=0, vmax=n)
# ax1.scatter(r1[:,0], r1[:,1], c=np.arange(n), cmap='jet', vmin=0, vmax=n)
# ax2.scatter(r2[:,0], r2[:,1], c=np.arange(n), cmap='jet', vmin=0, vmax=n)
# [ax.axis('equal') for ax in AX]
# plt.show()


# #(1) Find optimum-order correspondence (one dataset):
# dirREPO   = unipath.Path( os.path.dirname(__file__) ).parent
# name      = 'Bell'
# fname0    = os.path.join(dirREPO, 'Data', name, 'geom_sro.csv')
# a         = np.loadtxt(fname0, delimiter=',', skiprows=1)
# shape     = np.asarray(a[:,0], dtype=int)
# xy        = a[:,1:]
# r0        = [xy[shape==u]  for u in np.unique(shape)]
# nn        = [rr.shape[0]  for rr in r0]
# n         = max( nn )
# rtemp     = r0[ np.argmax(nn) ]
# r1        = [set_npoints(rr, n) for rr in r0]
# r1        = [corresp_roll(rtemp, rr)  for rr in r1]
# ### check ordering:
# plt.close('all')
# fig,AX  = plt.subplots( 2, 5, figsize=(12,6) )
# for ax,rr in zip(AX.flatten(), r1):
# 	ax.scatter(rr[:,0], rr[:,1], c=np.arange(rr.shape[0]), cmap='jet', vmin=0, vmax=n)
# 	ax.axis('equal')
# plt.show()



#(2) Process all datasets:
dirREPO   = unipath.Path( os.path.dirname(__file__) ).parent
names     = ['Bell', 'Comma', 'Device8',    'Face', 'Flatfish', 'Hammer',    'Heart', 'Horseshoe', 'Key']
for name in names:
	print( f'Finding roll correspondence for {name} dataset...' )
	fname0    = os.path.join(dirREPO, 'Data', name, 'geom_sro.csv')
	fname1    = os.path.join(dirREPO, 'Data', name, 'geom_sroc.csv')
	a         = np.loadtxt(fname0, delimiter=',', skiprows=1)
	shape     = np.asarray(a[:,0], dtype=int)
	xy        = a[:,1:]
	r0        = [xy[shape==u]  for u in np.unique(shape)]
	nn        = [rr.shape[0]  for rr in r0]
	n         = max( nn )
	rtemp     = r0[ np.argmax(nn) ]
	r1        = [set_npoints(rr, n) for rr in r0]
	r1        = [corresp_roll(rtemp, rr)  for rr in r1]
	write_csv(fname1, r1)



