import matplotlib.axes
import matplotlib.figure
from inspect import isfunction, signature
from mpl_toolkits.axes_grid1 import Divider, Size

class Box:

	def __init__(self, figure):
		self.figure = figure
		self.children = []
		self.showing_annotations = False
		self.post_update_functions = []

	def __iter__(self):
		return self.children.__iter__()

	def __next__(self):
		return self.children.__next__()

	def offset_x(self):
		if not hasattr(self, '_offset_x'):
			self._offset_x = 0.0
		return self._offset_x

	def offset_y(self):
		if not hasattr(self, '_offset_y'):
			self._offset_y = 0.0
		return self._offset_y

	def edge_left_x(self):
		return self.offset_x()

	def edge_right_x(self):
		return max([child.edge_right_x() for child in self.children])

	def edge_bottom_y(self):
		return self.offset_y()

	def edge_top_y(self):
		return max([child.edge_top_y() for child in self.children])

	def edge_center_x(self):
		return (self.edge_left_x() + self.edge_right_x())/2

	def edge_center_y(self):
		return (self.edge_bottom_y() + self.edge_top_y())/2

	def add_child(self, child):
		self.children.append(child)
		self.configure_children()
		self.check_annotations()
		self.figure.clip()

	def random_data(self, color=None):
		for child in self.children:
			child.random_data(color=color)

	def set_child_sizes(self, width=None, height=None):
		for child in self.children:
			if type(child) is matplotlib.axes.Axes:
				child.set_size(width=width, height=height)
			elif issubclass(type(child), Box):
				child.set_child_sizes(width=width, height=height)
		self.configure_children()
		self.check_annotations()
		self.figure.clip()

	def set_child_spacings(self, spacings=None, left=None, right=None, top=None, bottom=None):
		for child in self.children:
			if type(child) is matplotlib.axes.Axes:
				child.set_spacings(spacings=spacings, left=left, right=right, top=top, bottom=bottom)
			elif issubclass(type(child), Box):
				child.set_child_spacings(spacings=spacings, left=left, right=right, top=top, bottom=bottom)
		self.configure_children()
		self.check_annotations()
		self.figure.clip()

	def annotations(self):
		self.showing_annotations = True
		for child in self.children:
			if type(child) is matplotlib.axes.Axes:
				child.annotations()
			elif issubclass(type(child), Box):
				child.annotations()

	def clear_annotations(self):
		self.showing_annotations = False
		for child in self.children:
			if type(child) is matplotlib.axes.Axes:
				child.clear_annotations()
			elif issubclass(type(child), Box):
				child.clear_annotations()

	def check_annotations(self, show=False):
		if self.showing_annotations == True:
			self.annotations()
		else:
			self.clear_annotations()

	def animate(self, x, y):
		self._offset_x = x
		self._offset_y = y
		self.configure_children()
		self.figure.clip()

class HBox(Box):

	def configure_children(self):
		offset_x = self.offset_x
		offset_y = self.offset_y
		for child in self.children:
			child.offset_x = offset_x
			child.offset_y = offset_y
			offset_x = child.edge_right_x
			if issubclass(type(child), Box):
				child.configure_children()
		for fun in self.post_update_functions:
			fun()

class VBox(Box):

	def add_child(self, child):
		self.children.insert(0, child)
		self.configure_children()
		self.figure.clip()

	def configure_children(self):
		offset_x = self.offset_x
		offset_y = self.offset_y
		for child in self.children:
			child.offset_x = offset_x
			child.offset_y = offset_y
			offset_y = child.edge_top_y
			if issubclass(type(child), Box):
				child.configure_children()
		for fun in self.post_update_functions:
			fun()

def ax(self, width=2.4, height=2.4, spacings=0.05, left=0.8, right=0.6, top=0.4, bottom=0.6):

		h = [Size.Fixed(spacings), Size.Fixed(width)]
		v = [Size.Fixed(spacings), Size.Fixed(height)]
		divider = Divider(f, (0, 0, 1, 1), h, v, aspect=False)
		ax = self.add_axes(
			divider.get_position(), 
			axes_locator=divider.new_locator(nx=1, ny=1)
		)
		ax._offset_x = 0
		ax._offset_y = 0
		ax.post_update_functions = []

		# Important to initialize spacings as something (e.g. 0.0)
		ax.adjust(width=width, height=height, spacings=spacings, left=left, right=right, top=top, bottom=bottom)

		return ax

def clip(self):

	for ax in self.axes:
		ax.adjust()
		for fun in ax.post_update_functions:
			fun()

	width, height = 0.0, 0.0
	widths = [ax.edge_right_x() for ax in self.axes]
	if len(widths):
		width = max(widths)

	heights = [ax.edge_top_y() for ax in self.axes]
	if len(heights):
		height = max(heights)

	self.set_size_inches(width, height)

def hbox(self, data=None):
	box = HBox(self)
	if type(data) is int:
		for _ in range(data):
			box.add_child(self.ax())
	elif hasattr(data, '__iter__'):
		for child in data:
			box.add_child(child)
	return box

def vbox(self, data=None):
	box = VBox(self)
	if type(data) is int:
		for _ in range(data):
			box.add_child(self.ax())
	elif hasattr(data, '__iter__'):
		for child in data:
			box.add_child(child)
	return box

print('Importing figure extensions:')

functions = {name: thing for (name, thing) in locals().items() if isfunction(thing) and thing not in [isfunction, signature]}

for name, f in functions.items():
		target = matplotlib.figure.Figure
		if hasattr(target, name):
				print(f'*WARNING: {target} has already attribute: {name}, skipping.')
		else:
				print(f'Extending {target} with .{name}{signature(f)}')
				setattr(target, name, f)