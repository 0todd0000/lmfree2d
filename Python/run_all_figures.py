'''
Run all figure scripts (fig_*.py) and save figures as PDF
'''


import os

dir0 = os.path.dirname( __file__ )
for root,dirs,filenames in os.walk(dir0):
	for filename in filenames:
		if filename.startswith('fig_'):
			print( f'Running {filename}...' )
			fpath = os.path.join(root, filename)
			os.system( f'python {fpath}')

