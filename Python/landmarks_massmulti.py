
'''
Conduct mass-multivariate two-sample testing of the the landmark data.

This script calculates the Hotelling's T2 statistic for each landmark,
then conducts nonparametric, permutation inference.
'''

import os,unipath
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import spm1d


def two_sample_mass_multivariate_test(r0, r1):
	# calculate test statistic values at each landmark:
	spm     = spm1d.stats.nonparam.hotellings2(r0, r1)
	# conduct nonparametric mass-multivariate inference:
	spmi    = spm.inference(0.05, iterations=-1)
	# assemble results:
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
# ### run nonparametric permutation test:
# res      = two_sample_mass_multivariate_test(r0, r1)
# print(res)



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
	### run nonparametric permutation test:
	res      = two_sample_mass_multivariate_test(r0, r1)
	results.append(res)
	print(name, res['zmax'], res['p'])
### save:
fname1   = os.path.join(dirREPO, 'Results', 'landmarks_massmulti.csv')
n        = max([res['z'].size  for res in results])
header   = 'T2max,T2crit,P,' + ','.join( ['T2-%d'%(i+1)  for i in range(n)] )
with open(fname1, 'w') as f:
	f.write(header + '\n')
	for res in results:
		zmax,zc,p,z = res['zmax'], res['zstar'], res['p'], res['z']
		fmt      = ('%.3f,'*(z.size+3))[:-1] + '\n'
		f.write( fmt % ((zmax,zc,p)+tuple(z)) )


