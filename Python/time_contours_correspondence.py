
'''
Find optimum correspondence between shapes
'''

import os,unipath,time
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






#(2) Process all datasets:
dirREPO   = unipath.Path( os.path.dirname(__file__) ).parent
names     = ['Bell', 'Comma', 'Device8',    'Face', 'Flatfish', 'Hammer',    'Heart', 'Horseshoe', 'Key']
dt0,dt1   = [], []
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
	t0        = time.time()
	r1        = [set_npoints(rr, n) for rr in r0]
	dt0.append( time.time() - t0 )
	
	t0        = time.time()
	r1        = [corresp_roll(rtemp, rr)  for rr in r1]
	dt1.append( time.time() - t0 )
	
dt0,dt1 = np.array(dt0), np.array(dt1)


print('\n\n\n')
print('Contour fitting:')
print('    Execution time (mean): %.6f' %dt0.mean())
print('    Execution time (SD):   %.6f' %dt0.std(ddof=1))
print()
print('Correspondence:')
print('    Execution time (mean): %.6f' %dt1.mean())
print('    Execution time (SD):   %.6f' %dt1.std(ddof=1))


