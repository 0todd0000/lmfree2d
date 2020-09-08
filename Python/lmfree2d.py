
'''
lmfree2d:
A collection of functions for conducting landmark-free statistical analysis
of 2D contours shapes.

Module containing functions used across scripts.

This is the only module in this repository. All other PY files are scripts.
'''

import os,unipath
from functools import wraps
import numpy as np
import scipy.spatial
from matplotlib import pyplot as plt
from geomdl import fitting
import networkx as nx
import pycpd
from sklearn.neighbors import NearestNeighbors
import spm1d



# the following RGB colors are used in various figures and plotting functions
colors = np.array([
	[177,139,187],
	[166,154,196],
	[132,118,181],
	[225,215,231],
	[252,227,205],
	[231,179,159],
	[213,160,104],
	[166,198,226],
	[134,167,202],
   ]) / 255



class _skip_template(object):
	'''
	Decorator class for skipping a template array.
	
	For a function call "g = f(a, b)", where a and b are NumPy arrays,
	this decorator will cause the function to skip processing and return b,
	if a and b are the same.
	'''

	def __init__(self, f):
		self.f = f

	def __call__(self, r, *args, **kwargs):
		if np.array_equal(r, args[0]):
			r1 = r.copy()
		else:
			r1 = self.f(r, *args, **kwargs)
		return r1



class _process_mulitple_contours(object):
	'''
	Decorator class for skipping a template array.
	
	For a function call "g = f(a)", where a is a NumPy array,
	this decorator will cause the function to process all elements along
	the first dimension of a, if a is not a 2D array.
	
	Notes:
	- If a is a 2D array, this decorator will not have any effect.
	- Otherwise this decorator will cause the function to iteratively
	  process all 2D arrays: a[0], a[1], ...
	'''

	def __init__(self, f):
		self.f = f

	def __call__(self, r, *args, **kwargs):
		if r.ndim in [1,3]:
			r1 = np.array([self.f(rr, *args, **kwargs)  for rr in r])
		else:
			r1 = self.f(r, *args, **kwargs)
		return r1



def _pvalue2str(p, latex=False):
	'''
	Convert a probability value to a string.
	
	- If p is less than 0.001, "< 0.001" will be returned.
	- Otherwise p will be formatted to 3 decimal points
	'''
	
	if latex:
		s = r'$p < 0.001$' if (p < 0.001) else (r'$p = %.3f$' %p)
	else:
		s = '< 0.001' if (p<0.001) else '%.3f' %p
	return s



class TwoSampleSPMResults(object):
	'''
	Class continaing statistical results for two-sample tests
	
	Attributes:
	
	* alpha : Type I error rate
	* m0 : mean contour for first group
	* m1 : mean contour for second group
	* p : probability of observing the given z_max value, if m0=m1, given the underlying shape variance
	* z : test statistic values (one for each contour point)
	* zc : critical test statistic value at alpha
	
	Properties:
	
	* T2_critical : (same as "zc")
	* T2_max : (same as "z_max")
	* n : number of contour points
	* npoints : (same as "n")
	* z_crticial: (same as "zc")
	* z_max : maximum test statistic value
	* zi: test statistic values, thresholded at zc
	
	Methods:
	
	* plot : plot the results including mean shapes, excursion set and p value
	* write_csv : write all attributes to CSV file
	'''
	
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
		s += '   p            = %s\n'   %_pvalue2str(self.p)
		return s
	
	@property
	def T2_critical(self):
		return self.zc
	@property
	def T2_max(self):
		return self.z.max()
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
	def zi(self):
		zi                 = self.z.copy()
		zi[self.z<self.zc] = np.nan
		return zi
		
	def plot(self, ax=None, offset=(0,0), poffset=(0,0), fc='0.65', ec='k', vmin=None, vmax=None):
		'''
		Plot statistical results.
		
		Below "m0" and "m1" are the mean shapes for the first and second groups, respectively.
	
		Arguments:
	
		* ax : a Matplotlib axes object (default: plt.gca() )
		* offset : position offset for mean shapes
		* poffset : position offset for p value text (relative to m0 centroid)
		* fc : face color for m0
		* ec : edge color for m1
		* vmin : minimum color value for excursion set (values below vmin will have the same color as vmin)
		* vmax : maximum color value for excursion set (values above vmax will have the same color as vmax)
		'''
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
		ax.text(x0.mean()+pxo, y0.mean()+pyo, _pvalue2str(self.p, latex=True), ha='center', size=12)
		ax.axis('equal')
		ax.axis('off')

	def write_csv(self, fname):
		'''
		Write results to CSV file.
		
		Arguments:
		
		* fname : file name (use a ".csv" extension)
		'''
		with open(fname, 'w') as f:
			f.write('Two Sample SPM Results\n')
			f.write('alpha = %.3f\n' %self.alpha)
			f.write('T2_critical = %.3f\n' %self.zc)
			f.write('p = %.3f\n' %self.p)
			f.write('X0,Y0,X1,Y1,T2\n')
			for (x0,y0),(x1,y1),zz in zip(self.m0, self.m1, self.z):
				f.write('%.6f,%.6f,%.6f,%.6f,%.3f\n' %(x0,y0,x1,y1,zz))



@_process_mulitple_contours
@_skip_template
def corresp_roll(r, r_template):
	'''
	Find contour point correspondence using a simple optimal roll search.
	
	Notes: 
	
	* Below the variables (m,n) represent the number of shapes and number of contour points, respectively.
	* All shapes must have the same n.
	* Below the terms "source" and "template" refer to changing and non-changing contours, respectfully.
	
	
	Inputs:
	
	* r : a single source contour as an (n,2) array or multiple source contours as an (m,n,2) array
	* r_template : the template contour as an (n,2) array
	
	Outputs:
	
	* Optimally rolled contour(s) with same array shape as r
	'''
	f       = [sse(r_template, np.roll(r, i, axis=0) )   for i in range(r.shape[0])]
	i       = np.argmin(f)
	return np.roll(r, i, axis=0)



def get_repository_path():
	'''
	Return the respoitory path relative to this file (lmfree2d.py).
	
	The repository path is the parent of the directory in which this file is saved.
	'''
	return unipath.Path( os.path.dirname(__file__) ).parent



def get_shape_with_most_points(r):
	'''
	Return the contour shape that has the largest number of points.
	
	If there are multiple shapes with the same (maximum) number of points, the first will be returned.
	
	Inputs:
	
	*r* : multiple contour shapes as an (m,) array or m-length list of (n,2) contour shapes
	
	Outputs:
	
	* r_max : an (n,2) array, the element of r that has the most number of points
	* n_max : the number of points in r_max
	* ind   : the index of r_max in r
	'''
	npoints    = [rr.shape[0]  for rr in r]
	ind        = np.argmax(npoints)
	return r[ind], max(npoints), ind



@_process_mulitple_contours
def order_points_clockwise(r, clockwise=True):
	'''
	Order a set of 2D points clockwise (default) or counterclockwise along the contour.
	
	Notes:
	
	* The points must be ordered before using this function.
	* This function will only check the CW/CCW point ordering, and reverse the ordering if necessary
	* See "reorder_points" for ordering a set of unordered points.
	
	Inputs:
	
	* r : a single contour as an (n,2) array or multiple contours as an (m,) or (m,n,2) array
	* clockwise : bool  True=clockwise, False=counterclockwise
	
	Outputs: 
	
	r_ordered : contour(s) with ordered points, same array shape as r

	References:
	
	* https://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
	'''
	if np.all(r[0] == r[-1]):
		r = r[:-1]

	x,y    = r.T
	s      = (x[1:] - x[:-1]) * (y[1:] + y[:-1])
	s      = s.sum() # s > 0 implies clockwise
	# print('Clockwise' if s else 'Counterclockwise')
	cw     = s > 0
	if cw==clockwise:
		r = r[::-1]
	return r



def plot_correspondence(ax, r1, r0, c0=None, c1=None, c2=None):
	'''
	Plot point correspondence between contour shapes.
	
	Inputs:
	
	* ax : a Matplotlib axes object (e.g. use "ax = pyplot.axes()" to create an axes object)
	* r1 : one source contour shape as an (n,2) array, or multiple source contour shapes as an (m,) or (m,n,2) array
	* r0 : the template contour as an (n,2) array
	* c0 : color of template points
	* c1 : color of source points
	* c2 : color of correspondence lines
	'''
	def _plot(ax, r0, r1, c0=None, c1=None, c2=None):
		c0 = 'k' if (c0 is None ) else c0
		c1 = colors[0] if (c1 is None ) else c1
		c2 = colors[2] if (c2 is None ) else c2
		h0 = ax.plot(r0[:,0], r0[:,1], 'o', color=c0, ms=1, zorder=1)[0]
		h1 = ax.plot(r1[:,0], r1[:,1], 'o', color=c1, ms=1, zorder=1)[0]
		h2 = ax.plot(r0[0,0], r0[0,1], 'o', color=c0, mfc='w', mew=2, ms=8, zorder=3)[0]
		h3 = ax.plot(r1[0,0], r1[0,1], 'o', color=c1, mfc='w', mew=2, ms=8, zorder=3)[0]
		for (x0,y0),(x1,y1) in zip(r0,r1):
			h4 = ax.plot([x0,x1], [y0,y1], '-', color=c2, lw=0.5, zorder=0)[0]
		return h0,h1,h2,h3,h4
	if r1.ndim == 2:
		h0,h1,h2,h3,h4 = _plot(ax, r1, r0, c0, c1, c2)
	else:
		x,y   = np.meshgrid(1.2 + np.arange(5), [1.2,0])
		for xx,yy,rr in zip(x.flatten(), y.flatten(), r1):
			h0,h1,h2,h3,h4 = _plot(ax, rr+[xx,yy], r0+[xx,yy], c0, c1, c2)
	ax.axis('equal')
	ax.axis('off')
	return h0,h1,h2,h3,h4



def plot_point_order(ax, r, cmap='jet'):
	'''
	Plot contour point order using a colormap to specify point order.
	
	Inputs:
	
	* ax : a Matplotlib axes object (e.g. use "ax = pyplot.axes()" to create an axes object)
	* r : multiple contour shapes as an (m,) array or m-length list of (n,2) contour shapes
	* cmap : colormap name (see Matplotlib colormap documentation)
	
	Reference:
	
	https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
	'''
	def _plot(r):
		ax.scatter(r[:,0], r[:,1], c=np.arange(r.shape[0]), cmap=cmap)
	if r.ndim == 2:
		_plot(r)
	else:
		x,y   = np.meshgrid( 1.2 * np.arange(5), [1.2,0])
		for xx,yy,rr in zip(x.flatten(), y.flatten(), r):
			_plot( rr+[xx,yy] )
	ax.axis('equal')
	ax.axis('off')



def plot_registration(ax, r, r_template=None):
	'''
	Plot multiple shapes in an overlapping manner to check registration quality.
	
	Inputs:
	
	* ax : a Matplotlib axes object (e.g. use "ax = pyplot.axes()" to create an axes object)
	* r : one contour shape as an (n,2) array, or multiple contour shapes as an (m,) or (m,n,2) array
	* r_template : the template contour as an (n,2) array
	'''
	if r.ndim == 2:
		r = [r]
	h0        = [ax.plot(rr[:,0], rr[:,1], 'ko', lw=0.5, zorder=0, ms=2)[0] for rr in r][0]
	x,y       = None, None
	if (r_template is None) and (r.ndim==3):
		x,y   = r.mean(axis=0).T
		label = 'Mean'
	else:
		x,y   = r_template.T
		label = 'Template'
	if x is not None:
		h1 = ax.plot(x, y, 'ro', ms=8, zorder=1)[0]
		ax.legend([h0,h1], ['Source', label])
	ax.axis('equal')
	ax.axis('off')



@_process_mulitple_contours
def random_roll(r):
	'''
	Randomly roll contour points. Rolling a contour will change its starting point.
	
	Inputs:
	
	* r : a single contour as an (n,2) array or multiple contours as an (m,) or (m,n,2) array
	
	Outputs: 
	
	r_rolled : array of rolled contour points, same array shape as r
	'''
	n = r.shape[0]  # number of points
	i = np.random.randint(1, n-1)
	return np.roll(r, i, axis=0)



def read_csv(filename):
	'''
	Read contour shapes from a CSV file.
	
	Notes:
	
	* The CSV file should have one header row (e.g. column labels: Shape, X, Y)
	* Starting from the second row, the CSV file must have three columns:
		* Column 1 : integer label identifying a contour shape
		* Column 2 : X coordinate of contour point
		* Column 3 : Y coordinate of contour point
	* If all contour shapes have the same number of points (n), an (m,n,2) array will be returned, where m is the number of shapes
	* Otherwise an (m,) array will be returned, where each element is an (n,2) array
	
	Inputs:
	
	* filename : full path to an output CSV file, formatted as described above
	
	Outputs:
	
	* r : multiple contours as an (m,) or (m,n,2) array
	'''
	a         = np.loadtxt(filename, delimiter=',', skiprows=1)
	shape     = np.asarray(a[:,0], dtype=int)
	xy        = np.array( a[:,1:] )
	return np.array(  [xy[shape==u]   for u in np.unique(shape)]  )



def read_csv_spm(filename):
	'''
	Read SPM results from a CSV file.
	
	Notes:
	
	* The CSV file is written from the output of "two_sample_test". For example:
	
		>>>  results = two_sample_test(r0, r1)
		>>>  results.write_csv( 'my_results.csv' )
	
	Inputs:
	
	* filename : full path to a CSV file that contains results from "two_sample_test"
	
	Outputs:
	
	* results : TwoSampleSPMResults object (refer to the lmfree2d.TwoSampleSPMResults class definition)
	'''
	with open(filename, 'r') as f:
		lines = f.readlines()
	alpha = float( lines[1].strip().split(' = ')[1] ) 
	zc    = float( lines[2].strip().split(' = ')[1] ) 
	p     = float( lines[3].strip().split(' = ')[1] ) 
	A     = np.array([s.strip().split(',')   for s in lines[5:]], dtype=float)
	m0    = A[:,:2]
	m1    = A[:,2:4]
	z     = A[:,4]
	return TwoSampleSPMResults(m0, m1, z, alpha, zc, p)



def read_landmarks_csv(filename):
	'''
	Read landmarks from a CSV file.
	
	Notes:
	
	* The CSV file should have one header row (e.g. column labels: Shape, Landmark, X, Y)
	* Starting from the second row, the CSV file must have four columns:
		* Column 1 : integer label identifying a shape
		* Column 2 : integer label identifying a landmark
		* Column 3 : X landmark coordinate
		* Column 4 : Y landmark coordinate
	* All shapes must have the same number of landmarks
	
	Inputs:
	
	* filename : full path to an output CSV file, formatted as described above
	
	Outputs:
	
	* landmarks : (m,n,2) array
	'''
	a         = np.loadtxt(filename, delimiter=',', skiprows=1)
	shape     = np.asarray(a[:,0], dtype=int)
	xy        = np.array( a[:,2:] )
	return np.array(  [xy[shape==u]   for u in np.unique(shape)]  )



@_process_mulitple_contours
@_skip_template
def register_cpd(r, r_template):
	'''
	Register multiple contours using the Coherent Point Drift (CPD) algorithm.
	
	Notes: 
	
	* Below the variables (m,n) represent the number of shapes and number of contour points, respectively.
	* Below the terms "source" and "template" refer to changing and non-changing contours, respectfully.
	* Source shapes do not necessarily have to have the same n.
	
	
	Inputs:
	
	* r : a single source contour as an (n,2) array or multiple source contours as an (m,) or (m,n,2) array
	* r_template : the template contour as an (n,2) array
	
	Outputs:
	
	* r_reg : registered contour(s) with same array shape as r
	
	References:
	
	* https://github.com/siavashk/pycpd
	* https://siavashk.github.io/2017/05/14/coherent-point-drift/
	'''
	reg     = pycpd.RigidRegistration(X=r_template, Y=r)
	reg.register()
	r_reg   = reg.TY
	return r_reg



@_process_mulitple_contours
@_skip_template
def register_procrustes(r, r_template):
	'''
	Register multiple contours using a Procrustes fit.
	
	Notes: 
	
	* Below the variables (m,n) represent the number of shapes and number of contour points, respectively.
	* Below the terms "source" and "template" refer to changing and non-changing contours, respectfully.
	* Contours do not necessarily have to have the same n.
	
	
	Inputs:
	
	* r : a single source contour as an (n,2) array or multiple source contours as an (m,) or (m,n,2) array
	* r_template : the template contour as an (n,2) array
	
	Outputs:
	
	* r_reg : registered contour(s) with same array shape as r
	
	References:
	
	https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.procrustes.html
	'''
	c       = r_template.mean(axis=0)
	s       = np.linalg.norm( r_template - c )  # scale
	_,b,_   = scipy.spatial.procrustes(r_template, r)
	r1      = s * b + c
	return r1



@_process_mulitple_contours
def reorder_points(points, optimum_order=False, ensure_clockwise=True):
	'''
	Order unordered points to form a continuous contour line
	
	Notes: 
	
	* Below the variables (m,n) represent the number of shapes and number of contour points, respectively.
	* Below the terms "source" and "template" refer to changing and non-changing contours, respectfully.
	* Contours do not necessarily have to have the same n.
	
	
	Inputs:
	
	* points : a single source contour as an (n,2) array or multiple source contours as an (m,) or (m,n,2) array
	* optimum_order : bool, whether or not to optimally order the points
	* ensure_clockwise : bool, whether or not to ensure that the points are ordered clockwise
	
	Outputs:
	
	* points_ordered : contour shape(s) with ordered points, same array shape as points
	
	References:
	
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



@_process_mulitple_contours
def set_npoints(r, n):
	'''
	Set the number of contour points.
	
	Notes: 
	
	* The new contour points are calculated using NURBS interpolation.
	* The new contour points will be spaced equally from parametric position 0 to 1 around the contour.
	
	
	Inputs:
	
	* r : a single source contour as an (n,2) array or multiple source contours as an (m,) or (m,n,2) array
	* n : int, desired number of contour points
	
	Outputs:
	
	* r_new : contour shape(s) with n points, as an (n,2) or (m,n,2) array
	
	References:
	
	https://nurbs-python.readthedocs.io/en/latest/module_fitting.html
	'''
	pcurve    = fitting.interpolate_curve(list(r), 3)
	pcurve.sample_size = n
	return np.asarray( pcurve.evalpts )



def set_matplotlib_rcparams():
	'''
	Set rc paramameters for Matplotlib.
	
	This function is needed only if you wish to replicate the paper's figures.
	
	References:
	
	* https://matplotlib.org/3.3.1/tutorials/introductory/customizing.html
	'''
	plt.rcParams['mathtext.fontset'] = 'stix'
	plt.rcParams['font.family']      = 'Arial'
	plt.rcParams['xtick.labelsize']  = 8
	plt.rcParams['ytick.labelsize']  = 8



@_process_mulitple_contours
def shuffle_points(r):
	'''
	Randomly shuffle contour points.
	
	Notes:
	
	* Shuffled points are not ordered along the contour.
	* This function is useful for testing robustness to arbitrary point ordering.
	
	Inputs:
	
	* r : a single contour as an (n,2) array or multiple contours as an (m,) or (m,n,2) array
	
	Outputs: 
	
	r_shuffled : array of shuffled contour points, same array shape as r
	'''
	n     = r.shape[0]  # number of points
	ind   = np.random.permutation(n)
	return r[ind]



def sse(r0, r1):
	'''
	Calculate pointwise sum-of-squared-error (SSE) between two contour shapes
	
	Inputs:
	
	* r0 : a single contour as an (n,2) array
	* r1 : a single contour as an (n,2) array
	
	Outputs: 
	
	sse_value : the sum-of-squared distances between contour points
	'''
	return (np.linalg.norm(r1-r0, axis=1)**2).sum()



def two_sample_test(r0, r1, alpha=0.05, parametric=True, iterations=-1):
	'''
	Conduct a two-sample test comparing two groups of contours.
	
	Notes:
	
	* This function conducts mass-multivariate statistical analysis using statistical parametric mapping (SPM)
	* The "iterations" keyword argument is used only if "parametric" is False
	* Setting "iterations" to -1 will conduct all possible permutations the data
	* As a rule-of-thumb, 10000 iterations is usually sufficient to achieve numerical stability
	* For small samples there may be less than 10000 unique permutations;  in this case it is advisable to conduct all possible permutations
	
	Inputs:
	
	* r0 : contours for one group as an (m,n,2) array
	* r1 : contours for a second group as an (m,n,2) array
	* alpha : float, Type I error rate (default: 0.05)
	* parametric : bool, whether to conduct parametric (True) or nonparametric (False) inference
	* iterations : int, number of iterations for nonparametric (permutation) inference; default: -1
	
	References:
	
	* Taylor, J. E., & Worsley, K. J. (2008). Random fields of multivariate test statistics, with applications to shape analysis. Annals of Statistics, 36(1), 1–27. http://doi.org/10.1214/009053607000000406)
	* Chung, M. K., Worsley, K. J., Nacewicz, B. M., Dalton, K. M., & Davidson, R. J. (2010). General multivariate linear modeling of surface shapes using SurfStat. NeuroImage, 53(2), 491–505. http://doi.org/10.1016/j.neuroimage.2010.06.032
	'''
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



def write_csv(filename, r):
	'''
	Write contour shapes as a CSV file
	
	Inputs:
	
	* filename : full path to an output CSV file
	* r : multiple contours as an (m,) or (m,n,2) array
	'''
	with open(filename, 'w') as f:
		f.write('Shape,X,Y\n')
		for i,rr in enumerate(r):
			for x,y in rr:
				f.write('%d,%.6f,%.6f\n' %(i+1,x,y))