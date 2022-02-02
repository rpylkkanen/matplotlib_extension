# matplotlib_extension

Extending matplotlib classes with useful functions.

```python
import matplotlib.pyplot
import matplotlib_extension
from string import ascii_lowercase as letters

fig, axes = matplotlib.pyplot.subplots(3, 3)
X = matplotlib.pyplot.imread('example_image.jpg')

for i, (ax, letter) in enumerate(zip(axes.flatten(), letters)):
	ax.imshow(X)
	z = 0.1 * i
	ax.zoom(z)
	ax.set_panel_label(f'{letter})')
	ax.set_top_label(f'.zoom({z:.1f})')
	ax.set_xticks([])
	ax.set_yticks([])
	ax.scalebar(1/10, 'um', 0.04)

fig.set_size_inches((5, 5))
fig.savefig('example.png', dpi=100)
```

![Example](https://github.com/rpylkkanen/matplotlib_extension/blob/master/example.png)
