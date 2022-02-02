import matplotlib.axes
from matplotlib_scalebar.scalebar import ScaleBar
from inspect import isfunction, signature

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

def break_spine(self, spine, color='k', d=0.015):

    kwargs = {
      'transform': self.transAxes, 
      'color': color, 
      'clip_on': False, 
      'linewidth': matplotlib.rcParams['axes.linewidth'],
    }

    ax_h, ax_w = self.bbox.height, self.bbox.width
    h_w = ax_h/ax_w
    w_h = ax_w/ax_h
    y_bottom = (-d, +d)
    y_top = 1 - d, 1 + d

    if spine in ['left', 'right', 'top', 'bottom']:

      self.spines[spine].set_visible(False)

      if spine == 'right':

        x = (1 - d*h_w, 1 + d*h_w)
        self.plot(x, y_bottom, **kwargs)
        self.plot(x, y_top, **kwargs)
        self.tick_params(
          axis='y',
          which='both',
          right=False,
          labelright=False,
        )
      
      elif spine == 'left':

        x = (-d*h_w, +d*h_w)
        self.plot(x, y_bottom, **kwargs)
        self.plot(x, y_top, **kwargs)
        self.tick_params(
          axis='y',
          which='both',
          left=False,
          labelleft=False,
        )


functions = {name: thing for (name, thing) in locals().items() if isfunction(thing) and thing not in [isfunction, signature]}

for name, f in functions.items():
  target = matplotlib.axes.Axes
  if hasattr(target, name):
    print(f'*WARNING: {target} has already attribute: {name}, skipping.')
  else:
    print(f'Extending {target} with .{name}{signature(f)}')
    setattr(target, name, f)