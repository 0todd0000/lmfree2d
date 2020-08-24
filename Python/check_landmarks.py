
'''

'''

import os,unipath
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patheffects
import pandas as pd


def load_geom_and_stack(fnameCSV):
	a         = np.loadtxt(fnameCSV, delimiter=',', skiprows=1)
	shape     = np.asarray(a[:,0], dtype=int)
	xy        = a[:,1:]
	return np.array( [xy[shape==u]  for u in np.unique(shape)] )



#(0) Load data:
dirREPO    = unipath.Path( os.path.dirname(__file__) ).parent
names      = ['Bell', 'Comma', 'Device8', 'Face',    'Flatfish', 'Hammer', 'Heart', 'Horseshoe', 'Key']
name       = names[8]
fnameXY    = os.path.join(dirREPO, 'Data', name, 'geom_original.csv')
fnameLM    = os.path.join(dirREPO, 'Data', name, 'landmarks.csv')
r          = load_geom_and_stack(fnameXY)
df         = pd.read_csv(fnameLM, sep=',')






#(1) Plot:
plt.close('all')
plt.figure(figsize=(14,5))
plt.get_current_fig_manager().window.move(0, 0)
axx = np.linspace(0, 1, 6)[:5]
axy = [0.5, 0]
axw = axx[1]-axx[0]
axh = axy[0]-axy[1]
AX  = np.array([[plt.axes([xx,yy,axw,axh])  for xx in axx] for yy in axy])

for i,(ax,rr) in enumerate(zip(AX.flatten(), r)):
	ax.fill(rr[:,0], rr[:,1], edgecolor='b', fill=False)
	x,y  = df[ df['Shape'] == (i+1) ].values[:,2:].T
	ax.plot(x, y, 'ro', ms=8)
	ax.axis('equal')
	ax.axis('off')

plt.show()


