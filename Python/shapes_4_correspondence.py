
'''
Find optimum correspondence between shapes
'''

import os,unipath
import numpy as np
from matplotlib import pyplot as plt
from geomdl import fitting





# def corresp_roll(r0, r1):
# 	np.roll(self.verts, n, axis=0)
#
# 	f       = []
# 	s       = self.copy()
# 	for i in range(self.nvert):
# 		s.roll_vertices(1)
# 		f.append( s.sse(template) )
# 	i       = np.argmin(f) + 1
# 	s.roll_vertices(i)
# 	return s,i,f


def set_npoints(r, n):
	pcurve    = fitting.interpolate_curve(list(r), 3)
	pcurve.sample_size = n
	return np.asarray( pcurve.evalpts )
	

def write_csv(fname, r):
	shape  = np.hstack([[i+1]*rr.shape[0]  for i,rr in enumerate(r)])
	with open(fname1, 'w') as f:
		f.write('Shape,X,Y\n')
		for s,(x,y) in zip(shape, np.vstack(r)):
			f.write('%d,%.6f,%.6f\n' %(s,x,y))




#(0) Find optimum-order correspondence between two shapes:
dirREPO   = unipath.Path( os.path.dirname(__file__) ).parent
name      = 'Comma'
fname0    = os.path.join(dirREPO, 'Data', name, 'geom_sro.csv')
a         = np.loadtxt(fname0, delimiter=',', skiprows=1)
shape     = np.asarray(a[:,0], dtype=int)
xy        = a[:,1:]
r0        = xy[shape==1]
r1        = xy[shape==4]
n         = max(r0.shape[0], r1.shape[0])
r0        = set_npoints(r0, n)
r1        = set_npoints(r1, n)
### check ordering:
plt.close('all')
fig,AX  = plt.subplots( 1, 2, figsize=(8,3) )
ax0,ax1 = AX.flatten()
ax0.scatter(r0[:,0], r0[:,1], c=np.arange(n), cmap='jet', vmin=0, vmax=n)
ax1.scatter(r1[:,0], r1[:,1], c=np.arange(n), cmap='jet', vmin=0, vmax=n)
# ax1.fill(r1[:,0], r1[:,1], fill=False)
ax0.axis('equal')
ax1.axis('equal')
plt.show()





# #(1) Order points for one dataset:
# dirREPO   = unipath.Path( os.path.dirname(__file__) ).parent
# names     = ['Bell', 'Comma', 'Device8',    'Face', 'Flatfish', 'Hammer',    'Heart', 'Horseshoe', 'Key']
# name      = names[5]
# fname0    = os.path.join(dirREPO, 'Data', name, 'geom_sr.csv')
# a         = np.loadtxt(fname0, delimiter=',', skiprows=1)
# shape     = np.asarray(a[:,0], dtype=int)
# xy        = a[:,1:]
# r0        = [xy[shape==u]  for u in np.unique(shape)]
# r1        = [reorder_points(rr, optimum_order=True)  for rr in r0]
# r1        = [order_points_clockwise(rr) for rr in r1]
# ### check ordering:
# plt.close('all')
# fig,AX  = plt.subplots( 1, 2, figsize=(8,3) )
# ax0,ax1 = AX.flatten()
# [ax0.fill(rr[:,0], rr[:,1], fill=False, edgecolor='0.7')  for rr in r0]
# [ax1.fill(rr[:,0], rr[:,1], fill=False, edgecolor='0.7')  for rr in r1]
# ax0.axis('equal')
# ax1.axis('equal')
# plt.show()




# #(2) Order points for all datasets:
# dirREPO   = unipath.Path( os.path.dirname(__file__) ).parent
# names     = ['Bell', 'Comma', 'Device8',    'Face', 'Flatfish', 'Hammer',    'Heart', 'Horseshoe', 'Key']
# for name in names:
# 	print( f'Ordering points for {name} dataset...' )
# 	fname0    = os.path.join(dirREPO, 'Data', name, 'geom_sr.csv')
# 	fname1    = os.path.join(dirREPO, 'Data', name, 'geom_sro.csv')
# 	a         = np.loadtxt(fname0, delimiter=',', skiprows=1)
# 	shape     = np.asarray(a[:,0], dtype=int)
# 	xy        = a[:,1:]
# 	for u in np.unique(shape):
# 		i     = shape==u
# 		r     = [xy[shape==u]  for u in np.unique(shape)]
# 		r     = [reorder_points(rr, optimum_order=True)  for rr in r]
# 		r     = [order_points_clockwise(rr) for rr in r]
# 		write_csv(fname1, r)

