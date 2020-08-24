
'''
Conduct mass-multivariate two-sample testing of the the landmark data.

This script calculates the Hotelling's T2 statistic for each landmark,
then conducts nonparametric, permutation inference.
'''

import os,unipath,time
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import spm1d


def two_sample_test(r0, r1, parametric=True):
	# calculate test statistic values at each landmark:
	if parametric:
		spmi    = spm1d.stats.hotellings2(r0, r1).inference(0.05)
		z       = spmi.z     # test statistic value for each landmark
		zstar   = spmi.zstar # critical test statistic
		zmax    = z.max()    # maximum test statistic value
		p       = spm1d.rft1d.T2.sf(z.max(), spmi.df, spmi.Q, spmi.fwhm, withBonf=True)
	else:
		spmi    = spm1d.stats.nonparam.hotellings2(r0, r1).inference(0.05, iterations=-1)
		z       = spmi.z     # test statistic value for each landmark
		zstar   = spmi.zstar # critical test statistic
		zmax    = z.max()    # maximum test statistic value
		pdf     = spmi.PDF0  # permutation distribution
		p       = ( pdf >= zmax).mean()  # p value (percentage of values in pdf greater than or equal to T2max)
	return dict(zmax=zmax, zstar=zstar, p=p, z=z)
	



# #(0) Conduct mass multivariate two-sample test for one dataset:
# ### load data:
# dirREPO  = unipath.Path( os.path.dirname(__file__) ).parent
# name     = 'Bell'
# fname    = os.path.join(dirREPO, 'Data', name, 'landmarks_gpa.csv')
# df       = pd.read_csv(fname, sep=',')
# ### convert to 3D array (nshapes, nlandmarks, 2)
# nshapes  = df['SHAPE'].max()
# nlm      = df['LANDMARK'].max()
# r        = np.reshape( df[['X','Y']].values, (nshapes,nlm,2) )
# ### separate into groups:
# r0,r1    = r[:5], r[5:]
# ### run parametric test:
# t0       = time.time()
# res      = two_sample_test(r0, r1, parametric=True)
# dt       = time.time() - t0
# print(dt)
# ### run nonparametric permutation test:
# t0       = time.time()
# res      = two_sample_test(r0, r1, parametric=False)
# dt       = time.time() - t0
# print(dt)





#(1) Conduct mass multivariate two-sample test for all datasets:
dirREPO  = unipath.Path( os.path.dirname(__file__) ).parent
names    = ['Bell', 'Comma', 'Device8',    'Face', 'Flatfish', 'Hammer',    'Heart', 'Horseshoe', 'Key']
results  = []
for name in names:
	### load data:
	fname0   = os.path.join(dirREPO, 'Data', name, 'landmarks_gpa.csv')
	df       = pd.read_csv(fname0, sep=',')
	### convert to 3D array (nshapes, nlandmarks, 2)
	nshapes  = df['SHAPE'].max()
	nlm      = df['LANDMARK'].max()
	r        = np.reshape( df[['X','Y']].values, (nshapes,nlm,2) )
	### separate into groups:
	r0,r1    = r[:5], r[5:]
	### run parametric test:
	t0       = time.time()
	res      = two_sample_test(r0, r1, parametric=True)
	dt0      = time.time() - t0
	### run nonparametric permutation test:
	t0       = time.time()
	res      = two_sample_test(r0, r1, parametric=False)
	dt1      = time.time() - t0
	results.append([dt0,dt1])
	print(name, dt0, dt1)
### save:
fname1   = os.path.join(dirREPO, 'Results', 'time_landmarks_massmv.csv')
fmt      = '%s,%.6f,%.6f\n'
with open(fname1, 'w') as f:
	f.write('Name,t_param,t_nonparam\n')
	for name,(dt0,dt1) in zip(names,results):
		f.write( fmt % (name,dt0,dt1) )


