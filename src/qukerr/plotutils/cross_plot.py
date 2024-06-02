import numpy as np
import matplotlib.pyplot as plt

def cross_plot(*,
		xlabel = '$\\alpha$',
		ylabel = '$\\beta$' ,
		xlim   = [-8 , 8] ,
		ylim   = [-8 , 8] ,
		aspect = 'equal',
		ticks_frequency = 1,	
):
	
	try:
		plt.style.use('jlop')
	except:
		print('Could not find \'jlop\' matploblib theme. Falling back to Default')
	
	fig, ax = plt.subplots( figsize = (4,4))
	xmin, xmax = xlim
	ymin, ymax = ylim

	# Set identical scales for both axes
	ax.set(xlim=(xmin-1, xmax+1), ylim=(ymin-1, ymax+1), aspect=aspect)

   # Set bottom and left spines as x and y axes of coordinate system
	ax.spines['bottom'].set_position('zero')
	ax.spines['left'].set_position('zero')
	
	# Remove top and right spines
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)

	# Create 'x' and 'y' labels placed at the end of the axes
	ax.set_xlabel(xlabel, size=14, labelpad=-22, x=1.04)
	ax.set_ylabel(ylabel , size=14, labelpad=-20, y=1.02, rotation=0)

	# Create custom major ticks to determine position of tick labels
	x_ticks = np.arange(xmin, xmax+1, ticks_frequency)
	y_ticks = np.arange(ymin, ymax+1, ticks_frequency)
	ax.set_xticks(x_ticks[ ((x_ticks%2) != 1) & (x_ticks != 0) ])
	ax.set_yticks(y_ticks[ ((y_ticks%2) != 1) & (y_ticks != 0) ])

	# Create minor ticks placed at each integer to enable drawing of minor grid
	# lines: note that this has no effect in this example with ticks_frequency=1
	ax.set_xticks(np.arange(xmin, xmax+1), minor=True)
	ax.set_yticks(np.arange(ymin, ymax+1), minor=True)

	# Draw major and minor grid lines
	#ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)
	ax.grid(False)

	arrow_fmt = dict(markersize=4, color='black', clip_on=False)
	ax.plot((1), (0), marker='>', transform=ax.get_yaxis_transform(), **arrow_fmt)
	ax.plot((0), (1), marker='^', transform=ax.get_xaxis_transform(), **arrow_fmt)

	return fig, ax
