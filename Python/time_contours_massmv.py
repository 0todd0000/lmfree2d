
'''
Conduct mass-multivariate two-sample testing of the the landmark data.

This script calculates the Hotelling's T2 statistic for each landmark,
then conducts nonparametric, permutation inference.
'''

import os,unipath,time
import numpy as np
from matplotlib import pyplot as plt
import spm1d


def load_and_stack(fname):
	a         = np.loadtxt(fname0, delimiter=',', skiprows=1)
	shape     = np.asarray(a[:,0], dtype=int)
	xy        = a[:,1:]
	r         = np.array( [xy[shape==u]  for u in np.unique(shape)] )
	return r
	

def two_sample_test(r0, r1, parametric=True):
	if parametric:
		spm     = spm1d.stats.hotellings2(r0, r1).inference(0.05)
		z,zc    = spm.z, spm.zstar
		p       = spm1d.rft1d.T2.sf(z.max(), spm.df, spm.Q, spm.fwhm, withBonf=True)
	else:
		spm     = spm1d.stats.nonparam.hotellings2(r0, r1).inference(0.05, iterations=-1)
		z,zc    = spm.z, spm.zstar
		pdf     = spm.PDF0  # permutation distribution
		p       = ( pdf >= z.max()).mean()  # p value (percentage of values in pdf greater than or equal to T2max)
	zi        = z.copy()
	zi[z<zc]  = np.nan
	return z,zi,zc,p






#(1) Analyze all datasets:
dirREPO   = unipath.Path( os.path.dirname(__file__) ).parent
names     = ['Bell', 'Comma', 'Device8', 'Face',    'Flatfish', 'Hammer', 'Heart', 'Horseshoe', 'Key']
results   = []
for name in names:
	fname0    = os.path.join(dirREPO, 'Data', name, 'geom_sroc.csv')
	fname1a   = os.path.join(dirREPO, 'Data', name, 'spm.csv')
	fname1b   = os.path.join(dirREPO, 'Data', name, 'snpm.csv')
	r         = load_and_stack(fname0)
	r0,r1     = r[:5], r[5:]
	m         = r.mean(axis=0)
	
	t0        = time.time()
	z,zi,zc,p = two_sample_test(r0, r1, parametric=True)
	dt0       = time.time() - t0
	
	t0        = time.time()
	z,zi,zc,p = two_sample_test(r0, r1, parametric=False)
	dt1       = time.time() - t0
	print(name, dt0, dt1)
	results.append( [dt0,dt1] )
### save:
fname1   = os.path.join(dirREPO, 'Results', 'time_contours_massmv.csv')
fmt      = '%s,%.6f,%.6f\n'
with open(fname1, 'w') as f:
	f.write('Name,t_param,t_nonparam\n')
	for name,(dt0,dt1) in zip(names,results):
		f.write( fmt % (name,dt0,dt1) )


