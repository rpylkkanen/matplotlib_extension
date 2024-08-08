import matplotlib
import matplotlib.axes
from inspect import isfunction, signature
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import mpl_toolkits.mplot3d
from mpl_toolkits.axes_grid1 import Divider, Size
import numpy
import matplotlib.patheffects

def zoom(self, fraction):
		x1, x2 = fraction/2, 1 - fraction/2
		y1, y2 = fraction/2, 1 - fraction/2
		xmax = max(self.get_xlim())
		ymax = max(self.get_ylim())
		self.set_xlim(xmin=xmax*x1, xmax=xmax*x2)
		self.set_ylim(ymin=ymax*y2, ymax=ymax*y1)

def set_panel_label(self, text, size=10, weight='semibold'):
		# Is it possible to set this in rcparams?
		x = - self.left()/self.width()
		y = 1.0 + self.top()/self.height()
		ha = 'left'
		va = 'top'
		transform = self.transAxes
		self.text(x, y, text, ha=ha, va=va, weight=weight, transform=transform, size=size)

def set_top_label(self, text, kwargs={}):
		x = 0.5
		y = 1.0
		ha = 'center'
		va = 'bottom'
		transform = self.transAxes
		self.text(x, y, text, ha=ha, va=va, transform=transform, **kwargs)

def scalebar(self, x, text, height_inches=0.05, pad_inches=(0.05, 0.05), pad_data=(None, None), va='bottom', ha='center', color='#ffeb3b', font_size=9, text_lw=1, ec='k', weight='semibold'):
	
	fig = matplotlib.pyplot.gcf()
	
	xdata, ydata = pad_data
	if xdata == None:
		xdata = self.get_xlim()[0]
	if ydata == None:
		ydata = self.get_ylim()[0]
	
	t = matplotlib.transforms.blended_transform_factory(
			self.transData,
			fig.dpi_scale_trans + matplotlib.transforms.ScaledTranslation(xdata, ydata, self.transData),
	)

	xf = numpy.ptp(self.get_xlim())/self.width()  
	xpad, ypad = pad_inches

	x = numpy.array([0, 1, 1, 0]) * x + xpad * xf + xdata/np.ptp(self.get_xlim())*self.width()*xf
	y = numpy.array([0, 0, 1, 1]) * height_inches + ypad
	self.fill(x, y, ec='k', transform=t, clip_on=False, fc=color)
	path_effects = [matplotlib.patheffects.withStroke(linewidth=text_lw, foreground=ec)]
	self.text(numpy.mean(x), height_inches + ypad, text, ha=ha, va=va, size=font_size, transform=t, color=color, path_effects=path_effects, weight=weight)

def break_spine(self, spine, aspect=1/1, color='k', d=0.015, top=True, bottom=True):

		if not hasattr(self, '_broken_axes'):
				self._broken_axes = {
					'left': {
						'lines': [], 
					},
					'right': {
						'lines': [],
					},
				}
		else:
				if self._broken_axes is not None:
						for line in self._broken_axes[spine]['lines']:
								if line in self._children:
										self._children.remove(line)
						self._broken_axes[spine]['lines'] = []

		if hasattr(self, 'get_width') and hasattr(self, 'get_height'):
			aspect = self.get_height()/self.get_width()
		elif hasattr(self, 'width') and hasattr(self, 'height'):
			aspect = self.height()/self.width()
		else:
			aspect = self.bbox.height/self.bbox.width

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
						if top:
							line1 = self.add_artist(Line2D(
								x,
								y_top,
								**kwargs,
							))
							lines.append(line1)
						if bottom:
							line2 = self.add_artist(Line2D(
								x,
								y_bottom,
								**kwargs
							))
							lines.append(line2)

				self._broken_axes[spine]['lines'] = lines


def set_width(self, width, adjust=True):
	if width is not None:
		self._width = width
		if adjust:
			self.adjust()

def set_height(self, height, adjust=True):
	if height is not None:
		self._height = height
		if adjust:
			self.adjust()

def set_size(self, width=None, height=None, adjust=True):
	self.set_width(width, adjust=False)
	self.set_height(height, adjust=False)
	if adjust:
		self.adjust()

def set_left(self, left, adjust=True):
	if left is not None:
		self._left = left
		if adjust:
			self.adjust()

def set_right(self, right, adjust=True):
	if right is not None:
		self._right = right
		if adjust:
			self.adjust()

def set_top(self, top, adjust=True):
	if top is not None:
		self._top = top
		if adjust:
			self.adjust()

def set_bottom(self, bottom, adjust=True):
	if bottom is not None:
		self._bottom = bottom
		if adjust:
			self.adjust()

def set_spacings(self, spacings=None, left=None, right=None, top=None, bottom=None, adjust=True):
	
	self.set_left(spacings, adjust=False)
	self.set_right(spacings, adjust=False)
	self.set_top(spacings, adjust=False)
	self.set_bottom(spacings, adjust=False)

	if left is not None:
		self.set_left(left, adjust=False)
	if right is not None:
		self.set_right(right, adjust=False)
	if top is not None:
		self.set_top(top, adjust=False)
	if bottom is not None:
		self.set_bottom(bottom, adjust=False)

	if adjust:
		self.adjust()

def width(self):
	width = self.bbox.width
	if hasattr(self, '_width'):
		width = self._width
	return width

def height(self):
	height = self.bbox.height
	if hasattr(self, '_height'):
		height = self._height
	return height

def left(self):
	left = None
	if hasattr(self, '_left'):
		left = self._left
	return left

def right(self):
	right = None
	if hasattr(self, '_right'):
		right = self._right
	return right

def top(self):
	top = None
	if hasattr(self, '_top'):
		top = self._top
	return top

def bottom(self):
	bottom = None
	if hasattr(self, '_bottom'):
		bottom = self._bottom
	return bottom

def annotate_text(self):

	self.clear_text_annotations()

	w, h = self.width(), self.height()
	l, r, t, b = self.left(), self.right(), self.top(), self.bottom()

	data = dict(
		x={},
		y={},
		text={},
		rotation={}
	)
	
	data['x']['left'] = 0.0 - (l/w/2)
	data['x']['right'] = 1.0 + (r/w/2)
	data['x']['top'] = 0.5
	data['x']['bottom'] = 0.5
	data['x']['center'] = 0.5

	data['y']['left'] = 0.5
	data['y']['right'] = 0.5
	data['y']['top'] = 1.0 + (t/h/2)
	data['y']['bottom'] = 0.0 - (b/h/2)
	data['y']['center'] = 0.5

	data['text']['left'] = f'{l:.3f}"'
	data['text']['right'] = f'{r:.3f}"'
	data['text']['top'] = f'{t:.3f}"'
	data['text']['bottom'] = f'{b:.3f}"'
	data['text']['center'] = '\n'.join([f'w: {w:.3f}"', f'h: {h:.3f}"'])
	
	data['rotation']['left'] = 90
	data['rotation']['right'] = 90
	data['rotation']['top'] = 0
	data['rotation']['bottom'] = 0
	data['rotation']['center'] = 0

	kwargs = dict(
		ha='center',
		va='center',
		color='k',
		fontsize='x-small',
		transform=self.transAxes,
		bbox=dict(fc='white', alpha=0.75)
	)

	texts = []
	for key in ['left', 'right', 'top', 'bottom', 'center']:
		x = data['x'][key]
		y = data['y'][key]
		text = data['text'][key]
		rotation = data['rotation'][key]
		t = self.text(x, y, text, rotation=rotation, **kwargs)
		texts.append(t)

	self.text_annotations = texts

def annotate_rect(self):

		self.clear_rect_annotations()

		data = dict(
			x={},
			y={}
		)

		data['x']['vertical'] = [0.5, 0.5]
		data['x']['horizontal'] = [0.0, 1.0]
		data['y']['vertical'] = [0.0, 1.0]
		data['y']['horizontal'] = [0.5, 0.5]

		kwargs = dict(
			transform=self.transAxes,
			alpha=0.5,
		)

		lines = []
		for key in ['vertical', 'horizontal']:
			x = data['x'][key]
			y = data['y'][key]
			line = Line2D(x, y, **kwargs)
			l = self.add_artist(line)
			lines.append(l)

		self.line_annotations = lines
		

		w, h = self.width(), self.height()
		l, r, t, b = self.left(), self.right(), self.top(), self.bottom()

		data = dict(
			x={},
			y={},
			width={},
			height={},
			color={},
		)

		data['x']['left'] = 0.0 - l/w
		data['x']['right'] = 1.0
		data['x']['top'] = 0.0
		data['x']['bottom'] = 0.0

		data['y']['left'] = 0.0
		data['y']['right'] = 0.0
		data['y']['top'] = 1.0
		data['y']['bottom'] = 0.0 - b/h

		data['width']['left'] = l/w
		data['width']['right'] = r/w
		data['width']['top'] = w/w
		data['width']['bottom'] = w/w

		data['height']['left'] = h/h
		data['height']['right'] = h/h
		data['height']['top'] = t/h
		data['height']['bottom'] = b/h

		data['color']['left'] = '#F16A70'
		data['color']['right'] = '#B1D877'
		data['color']['top'] = '#8CDCDA'
		data['color']['bottom'] = '#4D4D4D'

		kwargs = dict(
			transform=self.transAxes,
			alpha=0.95,
			clip_on=False,
			linewidth=0.0,
		)

		patches = []
		for key in ['left', 'right', 'top', 'bottom']:
			x = data['x'][key]
			y = data['y'][key]
			width = data['width'][key]
			height = data['height'][key]
			color = data['color'][key]
			rect = Rectangle((x, y), width=width, height=height, color=color, **kwargs)
			patch = self.add_patch(rect)
			patches.append(patch)

		self.rect_annotations = patches	

def clear_text_annotations(self):

	if not hasattr(self, 'text_annotations'):
		self.text_annotations = None

	if self.text_annotations:
		for text in self.text_annotations:
				if text in self.texts:
						self.texts.remove(text)
		self.text_annotations = None

def clear_rect_annotations(self):

	if not hasattr(self, 'rect_annotations'):
		self.rect_annotations = None

	if self.rect_annotations:
		for patch in self.rect_annotations:
				if patch in self.patches:
						self.patches.remove(patch)
		self.rect_annotations = None

	if not hasattr(self, 'line_annotations'):
		self.line_annotations = None

	if self.line_annotations:
		for line in self.line_annotations:
				if line in self._children:
						self._children.remove(line)
		self.line_annotations = None

def annotations(self):
		if not hasattr(self, 'showing_annotations'):
			self.showing_annotations = True
		self.annotate_text()
		self.annotate_rect()

def clear_annotations(self):
		if not hasattr(self, 'showing_annotations'):
			self.showing_annotations = False
		self.clear_text_annotations()
		self.clear_rect_annotations()

def check_annotations(self):
		if not hasattr(self, 'showing_annotations'):
			self.showing_annotations = False
		elif self.showing_annotations:
			self.annotations()

def offset_x(self):
	if not hasattr(self, '_offset_x'):
		self._offset_x = 0.0
	return self._offset_x

def offset_y(self):
	if not hasattr(self, '_offset_y'):
		self._offset_y = 0.0
	return self._offset_y

def axis_right_x(self):
	return self.left() + self.width() + self.offset_x()

def axis_left_x(self):
	return self.left() + self.offset_x()

def axis_center_x(self):
	return self.left() + self.width()/2 + self.offset_x()

def axis_bottom_y(self):
	return self.bottom() + self.offset_y()

def axis_top_y(self):
	return self.bottom() + self.height() + self.offset_y()

def axis_center_y(self):
	return self.bottom() + self.height()/2 + self.offset_y()

def edge_left_x(self):
	return self.offset_x()

def edge_right_x(self):
	return self.left() + self.width() + self.right() + self.offset_x()

def edge_bottom_y(self):
	return self.offset_y()

def edge_top_y(self):
	return self.bottom() + self.height() + self.top() + self.offset_y()

def edge_center_x(self):
	return (self.left() + self.width() + self.right())/2 + self.offset_x()

def edge_center_y(self):
	return (self.bottom() + self.height() + self.top())/2 + self.offset_y()

def adjust(self, width=None, height=None, spacings=None, left=None, right=None, top=None, bottom=None):

	self.set_size(width=width, height=height, adjust=False)
	self.set_spacings(spacings=spacings, left=left, right=right, top=top, bottom=bottom, adjust=False)

	left = self.left()
	width = self.width()
	bottom = self.bottom()
	height = self.height()

	offset_x = self.offset_x()
	offset_y = self.offset_y()

	# Position
	f = self.get_figure()
	h = [Size.Fixed(left + offset_x), Size.Fixed(width)]
	v = [Size.Fixed(bottom + offset_y), Size.Fixed(height)]
	divider = Divider(f, (0, 0, 1, 1), h, v, aspect=False)
	self.set_position(divider.get_position())
	self.set_axes_locator(divider.new_locator(nx=1, ny=1))

	for fun in self.post_update_functions:
		fun()

def random_data(self, color=None):
	n = 10
	x = numpy.linspace(0, 50, num=n)
	y = numpy.random.randn(1, n).flatten()
	self.plot(x, y, 'o-', mfc='white', color=color)


def hexagon(self, x, y, width=1.0, height=1.0, width_fill=1.0, height_fill=1.0, perspective=0.25, xpad=0.2, ypad=0.2, kwargs_top={}, kwargs_bottom={}):

	# Coords
	x1, x2, x3, x4 = x - width/2, x - width/4, x + width/4, x + width/2
	y1, y2, y3, y4, y5 = y - height, y - height/2, y, y + height/2, y + height
	
	_x = [x1, x2, x3, x4]

	# Fills
	self.fill_between(_x, [y4, y3 + height*perspective, y3 + height*perspective, y4], [y4, y5 - height*perspective, y5 - height*perspective, y4], **kwargs_top)

	self.fill_between(_x, [y4, y3 + height*perspective, y3 + height*perspective, y4], [y2, y1 + height*perspective, y1 + height*perspective, y2], **kwargs_bottom)

	
	# Top     /^^\
	# self.plot(x, [y4, y5 - height*perspective, y5 - height*perspective, y4], color=line_color, **kwargs)
	
	# Top     \__/
	# self.plot(x, [y4, y3 + height*perspective, y3 + height*perspective, y4], color=line_color, **kwargs)
	
	# Bottom  \__/
	# self.plot(x, [y2, y1 + height*perspective, y1 + height*perspective, y2], color=line_color, **kwargs)
	
	# Center ||  ||
	# self.vlines([x1, x4], [y2, y2], [y4, y4], color=line_color, **kwargs)

	result = {
		'left_x': x1,
		'center_x': x,
		'right_x': x4,
		'left_center_y': numpy.average([y4, y2]),
		'left_top_y': y4,
		'left_bottom_y': y2,
		'right_center_y': numpy.average([y4, y2]),
		'right_top_y': y4,
		'right_bottom_y': y2,
		'top_y': y5 - height*perspective,
		'bottom_y': y1 + height*perspective,
		'width': width,
		'height': height,
	}

	return result

def cylinder(self, x0, y0, width, height, top_height=None, top_color="#e0e0e0", side_color="#9e9e9e", edge_color=None, linewidth=matplotlib.rcParams['axes.linewidth'], zorder=0):
	
	top_height = top_height or 0.25

	x = numpy.linspace(-1.0, 1.0, 1000)
	y = numpy.sqrt(1.0 ** 2 - x ** 2)
	x *= width/2
	x += x0
	y *= top_height

	y1 = y0 + y + height/2 - top_height
	y2 = y0 - y + height/2 - top_height
	y3 = y0 - y - height/2 + top_height

	self.fill_between(x, y2, y1, color=top_color, zorder=zorder, linewidth=linewidth, ec=edge_color)
	self.fill_between(x, y3, y2, color=side_color, zorder=zorder, linewidth=linewidth, ec=edge_color)
	self.plot(x, y1, '-', color=edge_color, zorder=zorder+0.5, lw=linewidth)
	self.plot(x, y2, '-', color=edge_color, zorder=zorder+0.5, lw=linewidth)
	self.plot(x, y3, '-', color=edge_color, zorder=zorder+0.5, lw=linewidth)
	self.plot([x[0]]*2, [y2[0], y3[0]], '-', color=edge_color, zorder=zorder+0.5, lw=linewidth)
	self.plot([x[-1]]*2, [y2[-1], y3[-1]], '-', color=edge_color, zorder=zorder+0.5, lw=linewidth)

def brace(self, xy1, xy2, s=None, r=0.05, text_pad=1.5, fontdict={}, **kwargs):

	scale = min([self.width(), self.height()])

	rx = r * numpy.ptp(self.get_xlim()) / (self.width() / scale)
	ry = r * numpy.ptp(self.get_ylim()) / (self.height() / scale)

	# calculate the angle
	theta = numpy.arctan2(xy2[1] - xy1[1], xy2[0] - xy1[0])

	# arc1 centre
	x11 = xy1[0] + rx * numpy.cos(theta)
	y11 = xy1[1] + ry * numpy.sin(theta)

	# arc2 centre
	x22 = (xy2[0] + xy1[0]) / 2.0 - 2.0 * rx * numpy.sin(theta) - rx * numpy.cos(theta)
	y22 = (xy2[1] + xy1[1]) / 2.0 + 2.0 * ry * numpy.cos(theta) - ry * numpy.sin(theta)

	# arc3 centre
	x33 = (xy2[0] + xy1[0]) / 2.0 - 2.0 * rx * numpy.sin(theta) + rx * numpy.cos(theta)
	y33 = (xy2[1] + xy1[1]) / 2.0 + 2.0 * ry * numpy.cos(theta) + ry * numpy.sin(theta)

	# arc4 centre
	x44 = xy2[0] - rx * numpy.cos(theta)
	y44 = xy2[1] - ry * numpy.sin(theta)

	# prepare the rotated
	q = numpy.linspace(theta, theta + numpy.pi/2.0, 50)

	# reverse q
	# t = np.flip(q) # this command is not supported by lower version of numpy
	t = q[::-1]

	# arc coordinates
	arc1x = rx * numpy.cos(t + numpy.pi/2.0) + x11
	arc1y = ry * numpy.sin(t + numpy.pi/2.0) + y11

	arc2x = rx * numpy.cos(q - numpy.pi/2.0) + x22
	arc2y = ry * numpy.sin(q - numpy.pi/2.0) + y22

	arc3x = rx * numpy.cos(q + numpy.pi) + x33
	arc3y = ry * numpy.sin(q + numpy.pi) + y33

	arc4x = rx * numpy.cos(t) + x44
	arc4y = ry * numpy.sin(t) + y44

	# plot arcs
	self.plot(arc1x, arc1y, **kwargs)
	self.plot(arc2x, arc2y, **kwargs)
	self.plot(arc3x, arc3y, **kwargs)
	self.plot(arc4x, arc4y, **kwargs)

	# plot lines
	self.plot([arc1x[-1], arc2x[1]], [arc1y[-1], arc2y[1]], **kwargs)
	self.plot([arc3x[-1], arc4x[1]], [arc3y[-1], arc4y[1]], **kwargs)

	if s is not None:
		rx = text_pad * rx
		ry = text_pad * ry
		x22 = (xy2[0] + xy1[0]) / 2.0 - 2.0 * rx * numpy.sin(theta) - rx * numpy.cos(theta)
		y22 = (xy2[1] + xy1[1]) / 2.0 + 2.0 * ry * numpy.cos(theta) - ry * numpy.sin(theta)
		arc2x = rx * numpy.cos(q - numpy.pi/2.0) + x22
		arc2y = ry * numpy.sin(q - numpy.pi/2.0) + y22
		text_x, text_y = arc2x[-1], arc2y[-1]
		rotation = numpy.degrees(theta) % 360.0
		self.text(text_x, text_y, s, rotation=rotation, rotation_mode='anchor', ha='center', va='center', fontdict=fontdict)

def attach(self, edge, target_ax, target_edge, clip=True):

		# Define mappings for edges to their corresponding methods
		edge_mappings = {
				'left': {'spacing': 'edge_left_x', 'axis': 'axis_left_x'},
				'right': {'spacing': 'edge_right_x', 'axis': 'axis_right_x'},
				'center': {'spacing': {'x': 'edge_center_x', 'y': 'edge_center_y'}, 
									 'axis': {'x': 'axis_center_x', 'y': 'axis_center_y'}},
				'top': {'spacing': 'edge_top_y', 'axis': 'axis_top_y'},
				'bottom': {'spacing': 'edge_bottom_y', 'axis': 'axis_bottom_y'},
		}

		# Split the edge and target_edge into direction and type
		edge_dir, edge_type = edge.split()
		target_dir, target_type = target_edge.split()

		# Determine if it's a horizontal or vertical attachment
		is_horizontal = 'left' in (edge_dir, target_dir) or 'right' in (edge_dir, target_dir)
		axis = 'x' if is_horizontal else 'y'

		# Handle center alignments
		if edge_dir == 'center' and target_dir == 'center':
				# Get methods for both axes
				target_x_method = getattr(target_ax, edge_mappings['center'][target_type]['x'])
				target_y_method = getattr(target_ax, edge_mappings['center'][target_type]['y'])
				
				if edge_type == 'spacing' and target_type == 'spacing':
						self.offset_x = lambda: target_x_method() - self.width()/2
						self.offset_y = lambda: target_y_method() - self.height()/2
				elif edge_type == 'axis' and target_type == 'axis':
						self.offset_x = lambda: target_x_method() - (self.left() + self.width()/2)
						self.offset_y = lambda: target_y_method() - (self.bottom() + self.height()/2)
				elif edge_type == 'spacing' and target_type == 'axis':
						self.offset_x = lambda: target_x_method() - (self.left() + self.width() + self.right())/2
						self.offset_y = lambda: target_y_method() - (self.bottom() + self.height() + self.top())/2
				elif edge_type == 'axis' and target_type == 'spacing':
						self.offset_x = lambda: target_x_method() - (self.left() + self.width()/2)
						self.offset_y = lambda: target_y_method() - (self.bottom() + self.height()/2)
				return lambda: None

		# Get the appropriate method for the target edge
		if target_dir == 'center':
				target_method = getattr(target_ax, edge_mappings[target_dir][target_type][axis])
		else:
				target_method = getattr(target_ax, edge_mappings[target_dir][target_type])

		# Set the offset based on the edge type and direction
		if is_horizontal:
				if edge_type == 'spacing':
						if edge_dir == 'left':
								self.offset_x = target_method
						elif edge_dir == 'right':
								self.offset_x = lambda: target_method() - (self.left() + self.width() + self.right())
						elif edge_dir == 'center':
								self.offset_x = lambda: target_method() - self.width()/2
				elif edge_type == 'axis':
						if edge_dir == 'left':
								self.offset_x = lambda: target_method() - self.left()
						elif edge_dir == 'right':
								self.offset_x = lambda: target_method() - (self.left() + self.width())
						elif edge_dir == 'center':
								self.offset_x = lambda: target_method() - (self.left() + self.width()/2)
		else:  # vertical
				if edge_type == 'spacing':
						if edge_dir == 'bottom':
								self.offset_y = target_method
						elif edge_dir == 'top':
								self.offset_y = lambda: target_method() - (self.bottom() + self.height() + self.top())
						elif edge_dir == 'center':
								self.offset_y = lambda: target_method() - self.height()/2
				elif edge_type == 'axis':
						if edge_dir == 'bottom':
								self.offset_y = lambda: target_method() - self.bottom()
						elif edge_dir == 'top':
								self.offset_y = lambda: target_method() - (self.bottom() + self.height())
						elif edge_dir == 'center':
								self.offset_y = lambda: target_method() - (self.bottom() + self.height()/2)

		# Return the attachment function for chaining
		if clip:
			f = self.get_figure()
			f.clip()
		return lambda: None

def attachkw(self, target_ax, line, clip=True):

	lrtbc = {'l': 'left', 'r': 'right', 't': 'top', 'b': 'bottom', 'c': 'center'}
	sa = {'s': 'spacing', 'a': 'axis'}

	n = 2
	chunks = [line[i:i+n] for i in range(0, len(line), n)]
	chunks = numpy.array(chunks).reshape(-1, 2)
	for edge, target_edge in chunks:
		edge = lrtbc[edge[0]] + ' ' + sa[edge[1]]
		target_edge = lrtbc[target_edge[0]] + ' ' + sa[target_edge[1]]
		self.attach(edge, target_ax, target_edge, clip=False)  
	if clip:	
		f = self.get_figure()
		f.clip()

functions = {name: thing for (name, thing) in locals().items() if isfunction(thing) and thing not in [isfunction, signature]}

for name, f in functions.items():
		targets = [matplotlib.axes.Axes]
		for target in targets:
			if hasattr(target, name):
					print(f'*WARNING: {target} has already attribute: {name}, skipping.')
			else:
					print(f'Extending {target} with .{name}{signature(f)}')
					setattr(target, name, f)


