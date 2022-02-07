import matplotlib.axes
from matplotlib_scalebar.scalebar import ScaleBar
from inspect import isfunction, signature
from matplotlib.lines import Line2D


def zoom(self, fraction):
    x1, x2 = fraction/2, 1 - fraction/2
    y1, y2 = fraction/2, 1 - fraction/2
    xmax = max(self.get_xlim())
    ymax = max(self.get_ylim())
    self.set_xlim(xmin=xmax*x1, xmax=xmax*x2)
    self.set_ylim(ymin=ymax*y2, ymax=ymax*y1)


def set_panel_label(self, text):
    x = 0.0
    y = 1.0
    ha = 'right'
    va = 'bottom'
    weight = 'semibold'
    transform = self.transAxes
    self.text(x, y, text, ha=ha, va=va, weight=weight, transform=transform)


def set_top_label(self, text):
    x = 0.5
    y = 1.0
    ha = 'center'
    va = 'bottom'
    transform = self.transAxes
    self.text(x, y, text, ha=ha, va=va, transform=transform)


def scalebar(self, dx, units, width_fraction, location='lower left', color='yellow'):
    bar = ScaleBar(
        dx,
        units,
        color=color,
        location=location,
        width_fraction=width_fraction,
        box_alpha=0.0,
    )
    self.add_artist(bar)
    return bar


def break_spine(self, spine, aspect=1/1, color='k', d=0.015):

    if not hasattr(self, '_broken_axes'):
        self._broken_axes = None
        self._broken_axes_aspect = None
        self._broken_axes_spine = None
        self._broken_axes_color = None
        self._broken_axes_d = None
    else:
        if self._broken_axes is not None:
            for line in self._broken_axes:
                if line in self._children:
                    self._children.remove(line)
            self._broken_axes = None
            self._broken_axes_aspect = None
            self._broken_axes_spine = None
            self._broken_axes_color = None
            self._broken_axes_d = None

    if hasattr(self, 'get_width') and hasattr(self, 'get_height'):
        aspect = self.get_height()/self.get_width()

    kwargs = {
      'transform': self.transAxes, 
      'color': color, 
      'clip_on': False, 
      'linewidth': matplotlib.rcParams['axes.linewidth'],
    }


    y_bottom = (-d, +d)
    y_top = 1 - d, 1 + d

    if spine in ['left', 'right', 'top', 'bottom']:

        self.spines[spine].set_visible(False)
        lines = []

        x = None

        if spine == 'right':

            x = (1 - d*aspect, 1 + d*aspect)

        elif spine == 'left':

            x = (-d * aspect, +d * aspect)
            self.set_yticks([])

        if x is not None:

            line1 = self.add_artist(Line2D(
              x,
              y_top,
              **kwargs,
            ))
            lines.append(line1)
            line2 = self.add_artist(Line2D(
              x,
              y_bottom,
              **kwargs
            ))
            lines.append(line2)

        self._broken_axes = lines
        self._broken_axes_aspect = aspect
        self._broken_axes_spine = spine
        self._broken_axes_color = color
        self._broken_axes_d = d

functions = {name: thing for (name, thing) in locals().items() if isfunction(thing) and thing not in [isfunction, signature]}

for name, f in functions.items():
    target = matplotlib.axes.Axes
    if hasattr(target, name):
        print(f'*WARNING: {target} has already attribute: {name}, skipping.')
    else:
        print(f'Extending {target} with .{name}{signature(f)}')
        setattr(target, name, f)
