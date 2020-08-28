
import os
import numpy as np
import lmfree2d as lm


#(0) Load data and summarize the number of contour points in each dataset:
dirREPO    = lm.get_repository_path()
names      = ['Bell', 'Comma', 'Device8', 'Face',    'Flatfish', 'Hammer', 'Heart', 'Horseshoe', 'Key']
npoints    = []
for name in names:
	fname0 = os.path.join(dirREPO, 'Data', name, 'contours.csv')
	r      = lm.read_csv(fname0)
	n      = np.asarray( [rr.shape[0]  for rr in r] )
	npoints.append( [n.min(), int( np.median(n) ), n.max()] )


#(1) Write the results:
fname1 = os.path.join(dirREPO, 'Results', 'table_npoints.csv')
with open(fname1, 'w') as f:
	f.write('Name,Min,Median,Max\n')
	for name,(mn,md,mx) in zip(names,npoints):
		f.write( f'{name},{mn},{md},{mx}\n')

