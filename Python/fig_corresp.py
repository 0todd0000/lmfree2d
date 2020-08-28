
import os,unipath
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
import networkx as nx

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family']      = 'Arial'
plt.rcParams['xtick.labelsize']  = 8
plt.rcParams['ytick.labelsize']  = 8

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



def corresp_roll(r0, r1):
	f       = []
	n       = r1.shape[0]
	for i in range(n):
		r   = np.roll(r1, i, axis=0)
		f.append( sse(r0, r) )
	i       = np.argmin(f)
	return np.roll(r1, i, axis=0), i

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


def sse(r0, r1):
	return (np.linalg.norm(r1-r0, axis=1)**2).sum()
	
def plot_registration(ax, r0, r1):
	h0 = ax.plot(r0[:,0], r0[:,1], 'o', color=c0, ms=1, zorder=1)[0]
	h1 = ax.plot(r1[:,0], r1[:,1], 'o', color=c1, ms=1, zorder=1)[0]
	h2 = ax.plot(r0[0,0], r0[0,1], 'o', color=c0, mfc='w', mew=2, ms=8, zorder=3)[0]
	h3 = ax.plot(r1[0,0], r1[0,1], 'o', color=c1, mfc='w', mew=2, ms=8, zorder=3)[0]
	for (x0,y0),(x1,y1) in zip(r0,r1):
		h4 = ax.plot([x0,x1], [y0,y1], '-', color=c2, lw=0.5, zorder=0)[0]
	ax.axis('equal')
	ax.axis('off')
	return h0,h1,h2,h3,h4




#(0) Load data:
dirREPO   = lm.get_repository_path()
names     = ['Bell', 'Comma', 'Device8', 'Face',    'Flatfish', 'Hammer', 'Heart', 'Horseshoe', 'Key']
name      = names[0]
fnameCSV  = os.path.join(dirREPO, 'Data', name, 'contours_sroc.csv')
a         = np.loadtxt(fnameCSV, delimiter=',', skiprows=1)
shape     = np.asarray(a[:,0], dtype=int)
xy        = a[:,1:]
i0,i1     = 1, 2
r0,r1     = xy[shape==i0], xy[shape==i1]


#(1) Process:
np.random.seed(0)
### shuffle:
n         = r0.shape[0]
ind0,ind1 = np.random.permutation(n), np.random.permutation(n)
rA0,rA1   = r0[ind0], r1[ind1]
### reorder:
rB0,rB1   = [reorder_points(r, optimum_order=True)  for r in [rA0,rA1]]
rB0,rB1   = [order_points_clockwise(r)  for r in [rB0,rB1]]
### optimum roll correspondence:
rD0       = rB0.copy()
rD1,i     = corresp_roll(rB0, rB1)
### intermediary roll:
rC0       = rB0.copy()
rC1       = np.roll(rB1, i-20, axis=0)




#(1) Plot:
plt.close('all')
plt.figure(figsize=(10,3))
# create axes:
axw,axh   = 0.25, 0.95
axx       = np.linspace(0, 1, 5)[:4]
AX        = [plt.axes([x,0,axw,axh])  for x in axx]
ax0,ax1,ax2,ax3 = AX
# plot:
c0,c1,c2  = 'k', colors[0], colors[2]
handles   = plot_registration(ax0, rA0, rA1)
plot_registration(ax1, rB0, rB1)
plot_registration(ax2, rC0, rC1)
plot_registration(ax3, rD0, rD1)
# panel labels:
[ax.text(0.52, 0.99, '(%s)' %chr(97+i), size=14, ha='center', transform=ax.transAxes)   for i,ax in enumerate(AX)]
labels    = ['Original', 'Ordered', 'Rolled', 'Optimum Roll']
[ax.text(0.5, 0.35, f'{s}', size=12, ha='center', transform=ax.transAxes, zorder=10, bbox=dict(facecolor='w', alpha=0.8))   for ax,s in zip(AX,labels)]
# legend:
labels    = ['Contour Point A', 'Contour Point B', 'Initial Point A', 'Initial Point B', 'Correspondence Line']
leg       = ax0.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.66, 0.6))
plt.setp(leg.get_texts(), size=8)

plt.show()




#(2) Save figure:
fnamePDF  = os.path.join(dirREPO, 'Figures', 'corresp.pdf')
plt.savefig(fnamePDF)


