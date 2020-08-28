
'''
Conduct mass multivariate (SPM) testing on two samples of contour shapes
'''

import os
import numpy as np
from matplotlib import pyplot as plt
import lmfree2d as lm



# #(0) Analyze a single dataset:
# dirREPO   = lm.get_repository_path()
# names     = ['Bell', 'Comma', 'Device8', 'Face',    'Flatfish', 'Hammer', 'Heart', 'Horseshoe', 'Key']
# name      = names[8]
# fname0    = os.path.join(dirREPO, 'Data', name, 'contours_sroc.csv')
# r         = lm.read_csv(fname0)
# r0,r1     = r[:5], r[5:]
# results   = lm.two_sample_test(r0, r1, parametric=True)
# # results   = lm.two_sample_test(r0, r1, parametric=False)
# print(results)
# ### plot:
# plt.close('all')
# plt.figure()
# ax        = plt.axes()
# results.plot(ax)
# ax.legend()
# plt.show()





#(1) Analyze all datasets:
dirREPO   = lm.get_repository_path()
names     = ['Bell', 'Comma', 'Device8', 'Face',    'Flatfish', 'Hammer', 'Heart', 'Horseshoe', 'Key']
for name in names:
	print( f'\nProcessing the {name} dataset (parametric)...' )
	fname0    = os.path.join(dirREPO, 'Data', name, 'contours_sroc.csv')
	fname1a   = os.path.join(dirREPO, 'Data', name, 'spm.csv')
	fname1b   = os.path.join(dirREPO, 'Data', name, 'snpm.csv')
	r         = lm.read_csv(fname0)
	r0,r1     = r[:5], r[5:]
	results   = lm.two_sample_test(r0, r1, parametric=True)
	results.write_csv(fname1a)
	print( f'Processing the {name} dataset (non-parametric)...' )
	results   = lm.two_sample_test(r0, r1, parametric=False)
	results.write_csv(fname1b)









