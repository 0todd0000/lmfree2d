
'''
Process all datasets and save results.

The results of all processing steps are saved for all datasets.

File names (all with extension: .csv):

contours      :  original contour point data
contours_s    :  shuffled points
contours_sr   :  registered
contours_sro  :  re-ordered points
contours_sroc :  points brought into correspondence
spm           :  parametric hypothesis testing results
snpm          :  nonparametric hypothesis testing results
'''


import os
import numpy as np
import lmfree2d as lm




#(0) Process all datasets:
dirREPO     = lm.get_repository_path()
names       = ['Bell', 'Comma', 'Device8',    'Face', 'Flatfish', 'Hammer',    'Heart', 'Horseshoe', 'Key']

for name in names:
	print(f'--- Processing {name} dataset ---')
	
	# specify file names:
	fname0      = os.path.join(dirREPO, 'Data', name, 'contours.csv')
	fname1      = os.path.join(dirREPO, 'Data', name, 'contours_s.csv')
	fname2      = os.path.join(dirREPO, 'Data', name, 'contours_sr.csv')
	fname3      = os.path.join(dirREPO, 'Data', name, 'contours_sro.csv')
	fname4      = os.path.join(dirREPO, 'Data', name, 'contours_sroc.csv')
	fnameSPM    = os.path.join(dirREPO, 'Data', name, 'spm.csv')
	fnameSnPM   = os.path.join(dirREPO, 'Data', name, 'snpm.csv')

	### load data
	r0          = lm.read_csv(fname0)
	
	### shuffle contour points:
	print('Shuffling contour points...')
	np.random.seed( 10 + names.index(name) )
	r1          = lm.shuffle_points(r0)
	lm.write_csv(fname1, r1)

	### CPD registration
	print('Registering using CPD...')
	rtemp0,n,i  = lm.get_shape_with_most_points(r1)
	r2          = lm.register_cpd(r1, rtemp0)
	lm.write_csv(fname2, r2)

	### reorder points:
	print('Re-ordering points...')
	r3          = lm.reorder_points(r2, optimum_order=True, ensure_clockwise=True)
	lm.write_csv(fname3, r3)

	### correspondence:
	print('Running correspondence algorithm...')
	rtemp1      = r3[i]
	r4          = lm.set_npoints(r3, n)
	r4          = lm.corresp_roll(r4, rtemp1)
	lm.write_csv(fname4, r4)

	### hypothesis test:
	print('Conducting statistical tests...')
	rA,rB       = r4[:5], r4[5:]
	results0    = lm.two_sample_test(rA, rB, parametric=True)
	results0.write_csv(fnameSPM)
	results1    = lm.two_sample_test(rA, rB, parametric=False)
	results1.write_csv(fnameSnPM)
	
	print('Done.\n\n\n')








