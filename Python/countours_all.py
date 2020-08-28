
'''
Shuffle the point order for all shapes
'''

import os,unipath
import numpy as np
from matplotlib import pyplot as plt
from geomdl import fitting
import pycpd
import spm1d


def corresp_roll(x0, x1):
	f       = []
	n       = x1.shape[0]
	for i in range(n):
		r   = np.roll(x1, i, axis=0)
		f.append( sse(x0, r) )
	i       = np.argmin(f)
	return np.roll(x1, i, axis=0)

def order_points_clockwise(verts, clockwise=True):
	'''
	Order a set of 2D points along the periphery in counterclockwise (or clockwise) order
	
	Reference:
	https://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
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


def register_cpd_single_pair(r0, r1):
	reg     = pycpd.RigidRegistration(X=r0, Y=r1)	
	reg.register()
	r1r     = reg.TY
	return r1r

def set_npoints(r, n):
	pcurve    = fitting.interpolate_curve(list(r), 3)
	pcurve.sample_size = n
	return np.asarray( pcurve.evalpts )

def sse(x0, x1):
	return (np.linalg.norm(x1-x0, axis=1)**2).sum()

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




def check_correspondence(ax, r0, r1):
	x0,y0     = r0.T
	x1,y1     = r1.T
	ax.plot(x0, y0, 'k.', ms=10)
	ax.plot(x1, y1, '.', color='0.7', ms=10)
	for xx0,xx1,yy0,yy1 in zip(x0,x1,y0,y1):
		ax.plot([xx0,xx1], [yy0,yy1], 'c-', lw=0.5)
	ax.axis('equal')




from sklearn.neighbors import NearestNeighbors
import networkx as nx


def reorder_points(points, optimum_order=False):
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
	return points[order]
	
	

#(0) Load data:
dirREPO   = lm.get_repository_path()
names     = ['Bell', 'Comma', 'Device8',    'Face', 'Flatfish', 'Hammer',    'Heart', 'Horseshoe', 'Key']
name      = names[8]
fname0    = os.path.join(dirREPO, 'Data', name, 'contours.csv')
a         = np.loadtxt(fname0, delimiter=',', skiprows=1)
shape     = np.asarray(a[:,0], dtype=int)
xy        = a[:,1:]
r0        = [xy[shape==u]  for u in np.unique(shape)]


#(1) Register:
npoints   = [rr.shape[0] for rr in r0]  # number of points for each shape
ind       = np.argmax(npoints)  # choose the shape with
# ind       = np.argmin(npoints)  # choose the shape with
# ind  = -1
rtemp     = r0[ind].copy()             # template
r1        = [register_cpd_single_pair(rtemp, rr)   for rr in r0]


#(2) Correspondence:
n         = max(npoints)
r2        = np.array(r1)
r2        = [reorder_points(rr, optimum_order=True)  for rr in r2]
r2        = np.array([set_npoints(rr, n) for rr in r2])
r2        = np.array([order_points_clockwise(rr) for rr in r2])
rtemp     = r2[ind].copy()
# rtemp     = r2.mean(axis=0)
r2        = np.array([corresp_roll(rtemp, rr)  for rr in r2])


#(3) SPM (mass multivariate test):
rA,rB     = r2[:5], r2[5:]
z,zi,zc,p = two_sample_test(rA, rB, parametric=True)
print(z.max(), zc, p)




#(4) Plot results:
plt.close('all')
fig,AX  = plt.subplots( 2, 2, figsize=(10,6) )
ax0,ax1,ax2,ax3 = AX.flatten()
[ax0.fill(rr[:,0], rr[:,1], edgecolor='0.7', fill=False)  for rr in r0]
[ax1.fill(rr[:,0], rr[:,1], edgecolor='0.7', fill=False)  for rr in r1]
# [ax2.fill(rr[:,0], rr[:,1], edgecolor='0.7', fill=False)  for rr in r2]

mA,mB = rA.mean(axis=0), rB.mean(axis=0)
[ax2.fill(rr[:,0], rr[:,1], edgecolor=cc, fill=False)   for rr,cc in zip([mA,mB],['k','r'])]

[ax.fill(rtemp[:,0], rtemp[:,1], color='c', fill=True, alpha=0.5)  for ax in [ax0,ax1]]

### SPM results:
m         = r2.mean(axis=0)
ax3.fill(m[:,0], m[:,1], color='0.7', zorder=1)
if np.any(z>zc):
	ax3.scatter(m[:,0], m[:,1], s=30, c=zi, cmap='hot', edgecolor='k', vmin=zc, vmax=z.max(), zorder=2)


[ax.axis('equal')  for ax in AX.flatten()]
[ax.axis('off')  for ax in AX.flatten()]
plt.show()


