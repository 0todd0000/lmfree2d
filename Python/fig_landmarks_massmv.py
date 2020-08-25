
'''
Plot results for mass-multivariate analysis of the the landmark data.
'''

import os,unipath
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


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


#(0) Load results:
dirREPO   = unipath.Path( os.path.dirname(__file__) ).parent
fnameCSV  = os.path.join(dirREPO, 'Results', 'landmarks_massmv.csv')
df        = pd.read_csv(fnameCSV, sep=',')
ndatasets = len(df)


#(1) Plot:
plt.close('all')
plt.figure( figsize=(10,7) )
axx       = np.linspace(0.05, 0.70, 3)
axy       = np.linspace(0.71, 0.06, 3)
axw,axh   = 0.27, 0.25
AX        = np.array([[plt.axes([x,y,axw,axh])   for x in axx]  for y in axy])

for i,ax in enumerate(AX.flatten()):
	row   = df.iloc[i]
	y     = np.array(row.values[4:], dtype=float)
	y     = y[np.logical_not(np.isnan(y))]
	x     = 1 + np.arange(y.size)
	ax.bar(x, y, color=colors[1])
	zc    = row['T2crit']
	ax.axhline(zc, color=colors[0], ls='--' )
	ax.text(5.7, zc+1.5, r'$T^2_{critical} = %.3f$'%zc, size=10, color=colors[0])
	ax.text(1, 30, r'$p = %.3f$'%row['P'], size=10)
	ax.set_title( row['Name'], color='k', size=16 )
[ax.set_xlabel('Landmark', color='k', size=14)  for ax in AX[2]]
[ax.set_ylabel('Test statistic value', color='k', size=14)  for ax in AX[:,0]]
plt.setp(AX[:2], xticklabels='')
plt.setp(AX[:,1:], yticklabels='')
plt.setp(AX, xlim=(0.5,8.5), ylim=(0,33))
plt.show()



#(2) Save figure:
fnamePDF  = os.path.join(dirREPO, 'Figures', 'landmarks_massmv.pdf')
plt.savefig(fnamePDF)
