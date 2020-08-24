
'''
Register contour points using the coherent point drift (CPD) algorithm
'''

import os,unipath,time
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




#(2) Register and save all datasets:
dirREPO   = unipath.Path( os.path.dirname(__file__) ).parent
names     = ['Bell', 'Comma', 'Device8',    'Face', 'Flatfish', 'Hammer',    'Heart', 'Horseshoe', 'Key']
dt        = []
for name in names:
	print( f'Registering {name} dataset...' )
	fname0    = os.path.join(dirREPO, 'Data', name, 'geom_s.csv')
	fname1    = os.path.join(dirREPO, 'Data', name, 'geom_sr.csv')
	a         = np.loadtxt(fname0, delimiter=',', skiprows=1)
	shape     = np.asarray(a[:,0], dtype=int)
	xy        = a[:,1:]
	t0        = time.time()
	xyr,_     = register_cpd_dataset(xy, shape)
	dt.append( time.time() - t0 )
dt = np.array(dt)


print('Execution time (mean): %.6f' %dt.mean())
print('Execution time (SD):   %.6f' %dt.std(ddof=1))
	
