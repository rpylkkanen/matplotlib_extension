import matplotlib.pyplot
import matplotlib_extension
from string import ascii_lowercase as letters

fig, axes = matplotlib.pyplot.subplots(3, 4)
X = matplotlib.pyplot.imread('example_image.jpg')

for i, (ax, letter) in enumerate(zip(axes.flatten(), letters)):
	if i < len(axes.flatten()) - 2: 
		ax.imshow(X)
		z = 0.1 * i
		ax.zoom(z)
		ax.set_panel_label(f'{letter})')
		ax.set_top_label(f'.zoom({z:.1f})')
		ax.set_xticks([])
		ax.set_yticks([])
		ax.scalebar(1/10, 'um', 0.04)
	elif i == len(axes.flatten()) - 2:
		spine = 'right'
		ax.plot([1, 2, 3], [1, 1.5, 2])
		ax.set_panel_label(f'{letter})')
		ax.break_spine(spine)
		ax.set_top_label(f'.break_spine\n({spine})')
	elif i == len(axes.flatten()) - 1:
		spine = 'left'
		ax.plot([4, 5, 6], [3, 3, 3])
		ax.set_panel_label(f'{letter})')
		ax.break_spine(spine)
		ax.set_top_label(f'.break_spine\n({spine})')

fig.set_size_inches((5, 5))
fig.savefig('example.png', dpi=100)



