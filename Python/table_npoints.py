
import os,unipath
import numpy as np



#(0) Load data:
dirREPO    = unipath.Path( os.path.dirname(__file__) ).parent
names      = ['Bell', 'Comma', 'Device8', 'Face',    'Flatfish', 'Hammer', 'Heart', 'Horseshoe', 'Key']
npoints    = []
for name in names:
	fname0 = os.path.join(dirREPO, 'Data', name, 'contours.csv')
	a      = np.loadtxt(fname0, delimiter=',', skiprows=1)
	shape  = np.asarray(a[:,0], dtype=int)
	n      = np.asarray(  [(shape==u).sum()  for u in np.unique(shape)],  dtype=int)
	npoints.append( [n.min(), int( np.median(n) ), n.max()] )


#(1) Write results:
fname1 = os.path.join(dirREPO, 'Results', 'table_npoints.csv')
with open(fname1, 'w') as f:
	f.write('Name,Min,Median,Max\n')
	for name,(mn,md,mx) in zip(names,npoints):
		f.write( f'{name},{mn},{md},{mx}\n')

