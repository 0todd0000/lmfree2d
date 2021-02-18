
'''
Process all datasets and save results
'''

import os
import numpy as np
from matplotlib import pyplot as plt
plt.ion()
import lmfree2d as lm



def process_data(r0):
	r0          = lm.shuffle_points(r0)
	### CPD registration
	rtemp0,n,i  = lm.get_shape_with_most_points(r0)
	r1          = lm.register_cpd(r0, rtemp0)
	### reorder points:
	r2          = lm.reorder_points(r1, optimum_order=True, ensure_clockwise=True)
	### correspondence:
	rtemp1      = r2[i]
	r3          = lm.set_npoints(r2, n)
	r4          = lm.corresp_roll(r3, rtemp1)
	return r4,rtemp1



#(0) Process data:
### load data
dirREPO     = lm.get_repository_path()
names       = ['Bell', 'Comma', 'Device8',    'Face', 'Flatfish', 'Hammer',    'Heart', 'Horseshoe', 'Key']
name        = names[6]
fname0      = os.path.join(dirREPO, 'Data', name, 'contours.csv')
r0          = lm.read_csv(fname0)
r0          = lm.shuffle_points(r0)
### CPD registration
rtemp0,n,i  = lm.get_shape_with_most_points(r0)
r1          = lm.register_cpd(r0, rtemp0)
### reorder points:
r2          = lm.reorder_points(r1, optimum_order=True, ensure_clockwise=True)
### correspondence:
rtemp1      = r2[i]
r3          = lm.set_npoints(r2, n)
r4          = lm.corresp_roll(r3, rtemp1)
### process data:
np.random.seed( 0 )  # Case 1
r1,rtemp1   = process_data(r0)
np.random.seed( 2 )  # Case 2
r2,rtemp2   = process_data(r0)
### hypothesis tests:
rA1,rB1     = r1[:5], r1[5:]
rA2,rB2     = r2[:5], r2[5:]
results1    = lm.two_sample_test(rA1, rB1, parametric=True)
results2    = lm.two_sample_test(rA2, rB2, parametric=True)




#(1) Plot all results:
plt.close('all')
lm.set_matplotlib_rcparams()
plt.figure( figsize=(10,8) )
axx,axy = [0.0, 0.5], [0.49, 0.0]
axw,axh = 0.5, 0.47
AX      = np.array([[plt.axes([x,y,axw,axh])  for x in axx]  for y in axy])
ax0,ax1,ax2,ax3 = AX.flatten()
# plot correspondence:
lm.plot_correspondence(ax0, rA1.mean(axis=0), rB1.mean(axis=0))
lm.plot_correspondence(ax1, rA2.mean(axis=0), rB2.mean(axis=0))
# plot hypothesis testing results:
results1.plot(ax2)
results2.plot(ax3)
# row labels:
labels = ['Correspondence', 'Hypothesis test']
[ax.text(0.01, 0.4, s, transform=ax.transAxes, rotation=90, size=16)  for ax,s in zip([ax0,ax2], labels)]
# column labels:
ax0.set_title('Case 1', size=16)
ax1.set_title('Case 2', size=16)
[ax.text(0.15, 1.01, '(%s)' %chr(97+i), size=13, transform=ax.transAxes)   for i,ax in enumerate(AX.flatten())]
ax2.legend()
plt.show()



#(2) Save figure:
fnamePDF  = os.path.join(dirREPO, 'Figures', 'sensitivity.pdf')
plt.savefig(fnamePDF)



