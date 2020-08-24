
'''
Find optimum correspondence between shapes
'''

import os,unipath
import numpy as np
from matplotlib import pyplot as plt
import spm1d

# colors = np.array([
# 	[177,139,187],
# 	[166,154,196],
# 	[132,118,181],
# 	[225,215,231],
# 	[252,227,205],
# 	[231,179,159],
# 	[213,160,104],
# 	[166,198,226],
# 	[134,167,202],
#    ]) / 255



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


def write_csv(fname, r, z, zc, p):
	A  = np.vstack([r.T, z]).T
	with open(fname, 'w') as f:
		f.write('SPM results\n')
		f.write('T2_critical = %.3f\n' %zc)
		f.write('p = %.3f\n' %p)
		# f.write('#Begin geometry\n')
		f.write('X,Y,T2\n')
		for (x,y),zz in zip(r, z):
			f.write('%.6f,%.6f,%.3f\n' %(x,y,zz))




# #(0) Analyze a single dataset:
# dirREPO   = unipath.Path( os.path.dirname(__file__) ).parent
# names     = ['Bell', 'Comma', 'Device8', 'Face',    'Flatfish', 'Hammer', 'Heart', 'Horseshoe', 'Key']
# name      = names[8]
# fname0    = os.path.join(dirREPO, 'Data', name, 'geom_sroc.csv')
# r         = load_and_stack(fname0)
# r0,r1     = r[:5], r[5:]
# z,zi,zc,p = two_sample_test(r0, r1, parametric=True)
# z,zi,zc,p = two_sample_test(r0, r1, parametric=False)
# print(p)
# ### plot:
# plt.close('all')
# plt.figure()
# ax        = plt.axes()
# m         = r.mean(axis=0)
# x,y       = m.T
# ax.fill(x, y, color='0.7', zorder=1)
# if np.any(z>zc):
# 	ax.scatter(x, y, s=30, c=zi, cmap='hot', edgecolor='k', vmin=zc, vmax=z.max(), zorder=2)
# ax.axis('equal')
# plt.show()





#(1) Analyze all datasets:
dirREPO   = unipath.Path( os.path.dirname(__file__) ).parent
names     = ['Bell', 'Comma', 'Device8', 'Face',    'Flatfish', 'Hammer', 'Heart', 'Horseshoe', 'Key']
for name in names:
	print( f'\nProcessing the {name} dataset (parametric)...' )
	fname0    = os.path.join(dirREPO, 'Data', name, 'geom_sroc.csv')
	fname1a   = os.path.join(dirREPO, 'Data', name, 'spm.csv')
	fname1b   = os.path.join(dirREPO, 'Data', name, 'snpm.csv')
	r         = load_and_stack(fname0)
	r0,r1     = r[:5], r[5:]
	m         = r.mean(axis=0)
	z,zi,zc,p = two_sample_test(r0, r1, parametric=True)
	write_csv(fname1a, m, z, zc, p)
	print( f'Processing the {name} dataset (non-parametric)...' )
	z,zi,zc,p = two_sample_test(r0, r1, parametric=False)
	write_csv(fname1b, m, z, zc, p)









