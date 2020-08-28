
'''
Shuffle the point order for all shapes
'''

import os,unipath
import numpy as np
from matplotlib import pyplot as plt
import lmfree2d as lm



#(0) Process a single dataset:
### load data
dirREPO     = lm.get_repository_path()
names       = ['Bell', 'Comma', 'Device8',    'Face', 'Flatfish', 'Hammer',    'Heart', 'Horseshoe', 'Key']
name        = names[4]
fname0      = os.path.join(dirREPO, 'Data', name, 'contours.csv')
r0          = lm.read_csv(fname0)
### CPD registration
rtemplate,n = lm.get_shape_with_most_points(r0)
r1          = lm.register_cpd_dataset(r0, rtemplate)
### correspondence:
r2          = lm.corresp_roll_dataset(r1, rtemplate)
### hypothesis test:
rA,rB       = r2[:5], r2[5:]
results     = lm.two_sample_test(rA, rB, parametric=True)
print(results)
### plot:
plt.close('all')
plt.figure()
ax        = plt.axes()
results.plot(ax)
ax.legend()
plt.show()



#
#
# #(4) Plot results:
# plt.close('all')
# fig,AX  = plt.subplots( 2, 2, figsize=(10,6) )
# ax0,ax1,ax2,ax3 = AX.flatten()
# [ax0.fill(rr[:,0], rr[:,1], edgecolor='0.7', fill=False)  for rr in r0]
# [ax1.fill(rr[:,0], rr[:,1], edgecolor='0.7', fill=False)  for rr in r1]
# # [ax2.fill(rr[:,0], rr[:,1], edgecolor='0.7', fill=False)  for rr in r2]
#
# mA,mB = rA.mean(axis=0), rB.mean(axis=0)
# [ax2.fill(rr[:,0], rr[:,1], edgecolor=cc, fill=False)   for rr,cc in zip([mA,mB],['k','r'])]
#
# [ax.fill(rtemp[:,0], rtemp[:,1], color='c', fill=True, alpha=0.5)  for ax in [ax0,ax1]]
#
# ### SPM results:
# m         = r2.mean(axis=0)
# ax3.fill(m[:,0], m[:,1], color='0.7', zorder=1)
# if np.any(z>zc):
# 	ax3.scatter(m[:,0], m[:,1], s=30, c=zi, cmap='hot', edgecolor='k', vmin=zc, vmax=z.max(), zorder=2)
#
#
# [ax.axis('equal')  for ax in AX.flatten()]
# [ax.axis('off')  for ax in AX.flatten()]
# plt.show()


