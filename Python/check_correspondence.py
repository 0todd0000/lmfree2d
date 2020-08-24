
'''

'''

import os,unipath
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def load_geom_and_stack(fnameCSV):
	a         = np.loadtxt(fnameCSV, delimiter=',', skiprows=1)
	shape     = np.asarray(a[:,0], dtype=int)
	xy        = a[:,1:]
	return np.array( [xy[shape==u]  for u in np.unique(shape)] )


def check_correspondence(ax, r0, r1):
	x0,y0     = r0.T
	x1,y1     = r1.T
	ax.plot(x0, y0, 'k.', ms=10)
	ax.plot(x1, y1, '.', color='0.7', ms=10)
	for xx0,xx1,yy0,yy1 in zip(x0,x1,y0,y1):
		ax.plot([xx0,xx1], [yy0,yy1], 'c-', lw=0.5)
	ax.axis('equal')



# #(0) Check correspondence for two shapes:
# dirREPO   = unipath.Path( os.path.dirname(__file__) ).parent
# name      = 'Bell'
# fname0    = os.path.join(dirREPO, 'Data', name, 'geom_sroc.csv')
# a         = np.loadtxt(fname0, delimiter=',', skiprows=1)
# shape     = np.asarray(a[:,0], dtype=int)
# xy        = a[:,1:]
# r0        = xy[shape==1]
# r1        = xy[shape==2]
# ### plot:
# plt.close('all')
# plt.figure()
# ax = plt.axes()
# check_correspondence(ax, r0, r1)
# plt.show()






#(1) Check correspondence for one dataset:
dirREPO   = unipath.Path( os.path.dirname(__file__) ).parent
names     = ['Bell', 'Comma', 'Device8',    'Face', 'Flatfish', 'Hammer',    'Heart', 'Horseshoe', 'Key']
name      = names[9]
fname0    = os.path.join(dirREPO, 'Data', name, 'geom_sroc.csv')
a         = np.loadtxt(fname0, delimiter=',', skiprows=1)
shape     = np.asarray(a[:,0], dtype=int)
xy        = a[:,1:]
r         = [xy[shape==u]  for u in np.unique(shape)]
### plot:
plt.close('all')
fig,AX = plt.subplots( 2, 5, figsize=(14,6) )
for i,ax in enumerate(AX.flatten()):
	check_correspondence(ax, r[0], r[i])

plt.show()