
'''
Register contour points using the coherent point drift (CPD) algorithm
'''

import os
import numpy as np
from matplotlib import pyplot as plt
import lmfree2d as lm






# #(0) Register one contour pair (single CPD iteration):
# dirREPO   = lm.get_repository_path()
# name      = 'Bell'
# fname0    = os.path.join(dirREPO, 'Data', name, 'contours_s.csv')
# r_all     = lm.read_csv(fname0)
# r0        = r_all[0]  # template shape
# r1        = r_all[1]  # source shape
# r1r       = lm.register_cpd_single_pair(r0, r1)
# ### check registration:
# plt.close('all')
# fig,AX  = plt.subplots( 1, 2, figsize=(8,3) )
# ax0,ax1 = AX.flatten()
# sc0     = ax0.scatter(r0[:,0], r0[:,1], color='b')
# sc1     = ax0.scatter(r1[:,0], r1[:,1], color='r')
# ax0.axis('equal')
# sc0     = ax1.scatter(r0[:,0], r0[:,1], color='b')
# sc1     = ax1.scatter(r1r[:,0], r1r[:,1], color='r')
# ax1.axis('equal')
# plt.show()



# #(1) Register one dataset:
# dirREPO    = lm.get_repository_path()
# name       = 'Bell'
# fname0     = os.path.join(dirREPO, 'Data', name, 'contours_s.csv')
# r0         = lm.read_csv(fname0)
# ### register:
# rtemplate  = lm.get_shape_with_most_points(r0)[0]
# r1         = lm.register_cpd_dataset(r0, rtemplate)
# ### check registration:
# plt.close('all')
# fig,AX     = plt.subplots( 1, 2, figsize=(8,3) )
# ax0,ax1    = AX.flatten()
# for rr in r0:
# 	ax0.scatter(rr[:,0], rr[:,1], color='0.7', s=3)
# for rr in r1:
# 	ax1.scatter(rr[:,0], rr[:,1], color='0.7', s=3)
# ax0.scatter(rtemplate[:,0], rtemplate[:,1], color='b', s=5)
# ax1.scatter(rtemplate[:,0], rtemplate[:,1], color='b', s=5)
# ax0.axis('equal')
# ax1.axis('equal')
# ax0.set_title('Original', size=14)
# ax1.set_title('Registered', size=14)
# plt.show()



#(2) Register and save all datasets:
dirREPO   = lm.get_repository_path()
names     = ['Bell', 'Comma', 'Device8',    'Face', 'Flatfish', 'Hammer',    'Heart', 'Horseshoe', 'Key']
for name in names:
	print( f'Registering {name} dataset...' )
	fname0    = os.path.join(dirREPO, 'Data', name, 'contours_s.csv')
	fname1    = os.path.join(dirREPO, 'Data', name, 'contours_sr.csv')
	r0        = lm.read_csv(fname0)
	rtemplate = lm.get_shape_with_most_points(r0)[0]
	r1        = lm.register_cpd_dataset(r0, rtemplate)
	lm.write_csv(fname1, r1)
