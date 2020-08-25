
'''
Order contour points in a clockwise manner
'''

import os,unipath
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
import networkx as nx


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



def write_csv(fname, r):
	shape  = np.hstack([[i+1]*rr.shape[0]  for i,rr in enumerate(r)])
	with open(fname1, 'w') as f:
		f.write('Shape,X,Y\n')
		for s,(x,y) in zip(shape, np.vstack(r)):
			f.write('%d,%.6f,%.6f\n' %(s,x,y))




# #(0) Order single set of contour points:
# dirREPO   = unipath.Path( os.path.dirname(__file__) ).parent
# name      = 'Comma'
# fname0    = os.path.join(dirREPO, 'Data', name, 'geom_sr.csv')
# a         = np.loadtxt(fname0, delimiter=',', skiprows=1)
# shape     = np.asarray(a[:,0], dtype=int)
# xy        = a[:,1:]
# r0        = xy[shape==1]
# r1        = reorder_points(r0, optimum_order=False)
# r1        = order_2d_points_clockwise(r1)
# print(r0.shape, r1.shape)
# ### check ordering:
# plt.close('all')
# fig,AX  = plt.subplots( 1, 2, figsize=(8,3) )
# ax0,ax1 = AX.flatten()
# ax0.fill(r0[:,0], r0[:,1], fill=False)
# ax1.fill(r1[:,0], r1[:,1], fill=False)
# ax0.axis('equal')
# ax1.axis('equal')
# plt.show()





# #(1) Order points for one dataset:
# dirREPO   = unipath.Path( os.path.dirname(__file__) ).parent
# names     = ['Bell', 'Comma', 'Device8',    'Face', 'Flatfish', 'Hammer',    'Heart', 'Horseshoe', 'Key']
# name      = names[5]
# fname0    = os.path.join(dirREPO, 'Data', name, 'geom_sr.csv')
# a         = np.loadtxt(fname0, delimiter=',', skiprows=1)
# shape     = np.asarray(a[:,0], dtype=int)
# xy        = a[:,1:]
# r0        = [xy[shape==u]  for u in np.unique(shape)]
# r1        = [reorder_points(rr, optimum_order=True)  for rr in r0]
# r1        = [order_points_clockwise(rr) for rr in r1]
# ### check ordering:
# plt.close('all')
# fig,AX  = plt.subplots( 1, 2, figsize=(8,3) )
# ax0,ax1 = AX.flatten()
# [ax0.fill(rr[:,0], rr[:,1], fill=False, edgecolor='0.7')  for rr in r0]
# [ax1.fill(rr[:,0], rr[:,1], fill=False, edgecolor='0.7')  for rr in r1]
# ax0.axis('equal')
# ax1.axis('equal')
# plt.show()




#(2) Order points for all datasets:
dirREPO   = unipath.Path( os.path.dirname(__file__) ).parent
names     = ['Bell', 'Comma', 'Device8',    'Face', 'Flatfish', 'Hammer',    'Heart', 'Horseshoe', 'Key']
for name in names:
	print( f'Ordering points for {name} dataset...' )
	fname0    = os.path.join(dirREPO, 'Data', name, 'contours_sr.csv')
	fname1    = os.path.join(dirREPO, 'Data', name, 'contours_sro.csv')
	a         = np.loadtxt(fname0, delimiter=',', skiprows=1)
	shape     = np.asarray(a[:,0], dtype=int)
	xy        = a[:,1:]
	for u in np.unique(shape):
		i     = shape==u
		r     = [xy[shape==u]  for u in np.unique(shape)]
		r     = [reorder_points(rr, optimum_order=True)  for rr in r]
		r     = [order_points_clockwise(rr) for rr in r]
		write_csv(fname1, r)

