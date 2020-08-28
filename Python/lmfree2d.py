
'''
Module containing functions used across scripts.

This is the only module in this repository. All other PY files are scripts.
'''

import os,unipath
import numpy as np
from matplotlib import pyplot as plt
from geomdl import fitting
import networkx as nx
import pycpd
from sklearn.neighbors import NearestNeighbors
import spm1d


def pvalue2str(p, latex=False):
	if latex:
		s = r'$p < 0.001$' if (p < 0.001) else (r'$p = %.3f$' %p)
	else:
		s = '< 0.001' if (p<0.001) else '%.3f' %p
	return s


class TwoSampleSPMResults(object):
	def __init__(self, m0, m1, z, alpha, zcrit, p):
		self.m0    = m0
		self.m1    = m1
		self.z     = z
		self.alpha = alpha
		self.zc    = zcrit
		self.p     = p
	
	def __repr__(self):
		s  = 'TwoSampleSPMResults\n'
		s += '   npoints      = %d\n'   %self.n
		s += '   T2_max       = %.3f\n' %self.z_max
		s += '----- Inference -----\n'
		s += '   alpha        = %.3f\n' %self.alpha
		s += '   T2_critical  = %.3f\n' %self.zc
		s += '   p            = %s\n'   %pvalue2str(self.p)
		return s
		
	
	@property
	def n(self):
		return self.z.size
	@property
	def npoints(self):
		return self.n
	@property
	def z_critical(self):
		return self.zc
	@property
	def z_max(self):
		return self.z.max()
	@property
	def T2_critical(self):
		return self.zc
	@property
	def T2_max(self):
		return self.z.max()
	@property
	def zi(self):
		zi                 = self.z.copy()
		zi[self.z<self.zc] = np.nan
		return zi
		
	def plot(self, ax=None, offset=(0,0), poffset=(0,0), fc='0.65', ec='k', vmin=None, vmax=None):
		ax = plt.gca() if (ax is None) else ax
		assert isinstance(ax, plt.Axes), '"ax" must be a Matplotlib Axes object'
		x0,y0    = (self.m0 + offset).T
		x1,y1    = (self.m1 + offset).T
		# plot mean contour shapes:
		ax.fill(x0, y0, color=fc, zorder=0, label='Mean A')
		ax.fill(x1, y1, edgecolor=ec, fill=False, zorder=1, label='Mean B')
		# plot suprathreshold points:
		if np.any( self.z > self.zc ):
			ax.scatter(x1, y1, s=30, c=self.zi, cmap='hot', edgecolor='k', vmin=vmin, vmax=vmax, zorder=2, label='Suprathreshold Points')
		# add p value as text:
		pxo,pyo = poffset
		ax.text(x0.mean()+pxo, y0.mean()+pyo, pvalue2str(self.p, latex=True), ha='center', size=12)
		ax.axis('equal')

	def write_csv(self, fname):
		with open(fname, 'w') as f:
			f.write('Two Sample SPM Results\n')
			f.write('T2_critical = %.3f\n' %self.zc)
			f.write('p = %.3f\n' %self.p)
			f.write('X0,Y0,X1,Y1,T2\n')
			for (x0,y0),(x1,y1),zz in zip(self.m0, self.m1, self.z):
				f.write('%.6f,%.6f,%.6f,%.6f,%.3f\n' %(x0,y0,x1,y1,zz))


def corresp_roll(r0, r1):
	f       = []
	n       = r1.shape[0]
	for i in range(n):
		r   = np.roll(r1, i, axis=0)
		f.append( sse(r0, r) )
	i       = np.argmin(f)
	return np.roll(r1, i, axis=0)


def get_repository_path():
	return unipath.Path( os.path.dirname(__file__) ).parent


def get_shape_with_most_points(r):
	npoints    = [rr.shape[0]  for rr in r]
	ind        = np.argmax(npoints)
	return r[ind], max(npoints)

def order_points_clockwise(verts, clockwise=True):
	'''
	Order a set of 2D points along the periphery in counterclockwise (or clockwise) order
	
	Reference:
	https://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
	
	INPUTS:
	
	verts : (N,2) numpy array of ordered vertex coordinates
	
	clockwise : bool  True=clockwise, False=counterclockwise
	
	OUTPUTS: 
	
	verts_ordered = (N,2) numpy array of re-ordered vertex indices
	'''
	if np.all(verts[0] == verts[-1]):
		verts = verts[:-1]

	x,y    = verts.T
	s      = (x[1:] - x[:-1]) * (y[1:] + y[:-1])
	s      = s.sum() # s > 0 implies clockwise
	# print('Clockwise' if s else 'Counterclockwise')
	cw     = s > 0
	if cw==clockwise:
		verts = verts[::-1]
	return verts


def random_roll(r):
	n = r.shape[0]  # number of points
	i = np.random.randint(1, n-1)
	return np.roll(r, i, axis=0)

def read_csv(filename):
	a         = np.loadtxt(filename, delimiter=',', skiprows=1)
	shape     = np.asarray(a[:,0], dtype=int)
	xy        = np.array( a[:,1:] )
	return np.array(  [xy[shape==u]   for u in np.unique(shape)]  )



def register_cpd_single_pair(r0, r1):
	reg     = pycpd.RigidRegistration(X=r0, Y=r1)	
	reg.register()
	r1r     = reg.TY
	return r1r


def register_cpd_dataset(r, r0):
	r = r.copy()
	for i,rr in enumerate(r):
		if np.all(rr == r0):
			continue
		r[i] = register_cpd_single_pair(r0, rr)
	return r
	
	



def reorder_points(points, optimum_order=False, ensure_clockwise=True):
	'''
	Sorting points to form a continuous line
	
	https://stackoverflow.com/questions/37742358/sorting-points-to-form-a-continuous-line
	'''
	points = np.asarray(points, dtype=float)
	clf    = NearestNeighbors(2, radius=0.05, algorithm='auto', leaf_size=4, metric='minkowski', p=4).fit(points)
	G      = clf.kneighbors_graph()
	T      = nx.from_scipy_sparse_matrix(G)
	order  = list(nx.dfs_preorder_nodes(T, None, None))
	if optimum_order:
		npoints = points.shape[0]
		paths = [list(nx.dfs_preorder_nodes(T, i)) for i in range(npoints)]
		mindist = np.inf
		minidx = 0
		for i in range( npoints ):
			p = paths[i]           # order of nodes
			if len(p) < (0.5 * npoints):
				continue
			ordered = points[p]    # ordered nodes
			# find cost of that order by the sum of euclidean distances between points (i) and (i+1)
			cost = (((ordered[:-1] - ordered[1:])**2).sum(1)).sum()
			if cost < mindist:
				mindist = cost
				minidx = i
		order = paths[minidx]
	points    = points[order]
	if ensure_clockwise:
		points = order_points_clockwise(points)
	return points


def set_npoints(r, n):
	pcurve    = fitting.interpolate_curve(list(r), 3)
	pcurve.sample_size = n
	return np.asarray( pcurve.evalpts )



def shuffle_points(r):
	n     = r.shape[0]  # number of points
	ind   = np.random.permutation(n)
	return r[ind]


def sse(r0, r1):
	return (np.linalg.norm(r1-r0, axis=1)**2).sum()


def two_sample_test(r0, r1, alpha=0.05, parametric=True, iterations=-1):
	if parametric:
		spm     = spm1d.stats.hotellings2(r0, r1).inference(alpha)
		z,zc    = spm.z, spm.zstar
		p       = spm1d.rft1d.T2.sf(z.max(), spm.df, spm.Q, spm.fwhm, withBonf=True)
	else:
		spm     = spm1d.stats.nonparam.hotellings2(r0, r1).inference(alpha, iterations=iterations)
		z,zc    = spm.z, spm.zstar
		pdf     = spm.PDF0  # permutation distribution
		p       = ( pdf >= z.max()).mean()  # p value (percentage of values in pdf greater than or equal to T2max)
	m0,m1       = r0.mean(axis=0), r1.mean(axis=0)
	return TwoSampleSPMResults(m0, m1, z, alpha, zc, p)
	# zi        = z.copy()
	# zi[z<zc]  = np.nan
	# return z,zi,zc,p


def write_csv(filename, r):
	with open(filename, 'w') as f:
		f.write('Shape,X,Y\n')
		for i,rr in enumerate(r):
			for x,y in rr:
				f.write('%d,%.6f,%.6f\n' %(i+1,x,y))