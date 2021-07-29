import math
import os
import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.patches import Circle, Ellipse


def subplot_layout(n):
    r = round(math.sqrt(n))
    c = math.ceil(n / r)
    return r, c


# ======================================================================================================================
# Colouring
# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------
# Parula colour map (IP of MathWorks - for illustration only - do not include in executable)
cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905],
           [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143],
           [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 0.779247619],
           [0.1252714286, 0.3242428571, 0.8302714286], [0.0591333333, 0.3598333333, 0.8683333333],
           [0.0116952381, 0.3875095238, 0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571],
           [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 0.8719571429],
           [0.0498142857, 0.4585714286, 0.8640571429], [0.0629333333, 0.4736904762, 0.8554380952],
           [0.0722666667, 0.4886666667, 0.8467], [0.0779428571, 0.5039857143, 0.8383714286],
           [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 0.8262714286],
           [0.0640571429, 0.5569857143, 0.8239571429], [0.0487714286, 0.5772238095, 0.8228285714],
           [0.0343428571, 0.5965809524, 0.819852381], [0.0265, 0.6137, 0.8135],
           [0.0238904762, 0.6286619048, 0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667],
           [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 0.7607190476],
           [0.0383714286, 0.6742714286, 0.743552381], [0.0589714286, 0.6837571429, 0.7253857143],
           [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429],
           [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 0.6424333333],
           [0.2178285714, 0.7250428571, 0.6192619048], [0.2586428571, 0.7317142857, 0.5954285714],
           [0.3021714286, 0.7376047619, 0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667],
           [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 0.5033142857],
           [0.4871238095, 0.7490619048, 0.4839761905], [0.5300285714, 0.7491142857, 0.4661142857],
           [0.5708571429, 0.7485190476, 0.4493904762], [0.609852381, 0.7473142857, 0.4336857143],
           [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333],
           [0.7184095238, 0.7411333333, 0.3904761905], [0.7524857143, 0.7384, 0.3768142857],
           [0.7858428571, 0.7355666667, 0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762],
           [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217],
           [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 0.2886428571],
           [0.9738952381, 0.7313952381, 0.266647619], [0.9937714286, 0.7454571429, 0.240347619],
           [0.9990428571, 0.7653142857, 0.2164142857], [0.9955333333, 0.7860571429, 0.196652381],
           [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857],
           [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309],
           [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 0.0948380952],
           [0.9661, 0.9514428571, 0.0755333333], [0.9763, 0.9831, 0.0538]]

cm_parula = LinearSegmentedColormap.from_list('parula', cm_data)


# ----------------------------------------------------------------------------------------------------------------------
# Default colouring scheme - first 7 from Matlab default, next 10 from Matplotlib default
DEFAULT_COLOURS = [np.array([0.,   114., 189.]) / 255.,
                   np.array([217., 83.,   25.]) / 255.,
                   np.array([237., 177.,  32.]) / 255.,
                   np.array([126., 47.,  142.]) / 255.,
                   np.array([119., 172.,  48.]) / 255.,
                   np.array([77.,  190., 238.]) / 255.,
                   np.array([162., 20.,   47.]) / 255.,
                   '#1f77b4',
                   '#ff7f0e',
                   '#2ca02c',
                   '#d62728',
                   '#9467bd',
                   '#8c564b',
                   '#e377c2',
                   '#7f7f7f',
                   '#bcbd22',
                   '#17becf']


def default_color(i):
    return DEFAULT_COLOURS[i % len(DEFAULT_COLOURS)]


# ======================================================================================================================
# Set fontsize of axes
# ======================================================================================================================
def set_font_sizes(ax, size=12):

    # title and axis labels
    items = [ax.title, ax.xaxis.label, ax.yaxis.label]

    # offset text (e.g. scientific notation)
    items += [ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()]

    # add z-axis texts if 3D
    if ax.name == '3d':
        items += [ax.zaxis.label, ax.zaxis.get_offset_text()]

    # tick labels
    items += ax.get_xticklabels() + ax.get_yticklabels()

    # legend
    legend = ax.get_legend()
    if legend is not None:
        items += legend.get_texts()

    for item in items:
        item.set_fontsize(size)


# ======================================================================================================================
# Circle constraints
# ======================================================================================================================
def hollow_circle(oi, ri, ou, ru, kwargs={}):
    """

    NOTE: https://matplotlib.org/examples/api/donut_demo.html

    Parameters
    ----------
    oi : array_like
        Origin of inner circle, array with (x, y).
    ri : float
        Radius of inner circle.
    ou : array_like
        Origin of outer circle, array with (x, y).
    ru : float
        Radius of outer circle.
    kwargs : dictionary
        Keyword arguments to matplotlib.patches.PathPatch

    Returns
    -------
    circle : Patch
        A matplotlib patch which can be passed to ax.add_patch().
    """

    # generate vertices with a given radius
    vi = _get_circle_vertices(ri)
    vu = _get_circle_vertices(ru)

    # translate vertices to desired origin
    vi[:, 0] += oi[0]
    vi[:, 1] += oi[1]
    vu[:, 0] += ou[0]
    vu[:, 1] += ou[1]

    # matplotlib sorcery
    codes = np.ones(len(vi), dtype=mpath.Path.code_type) * mpath.Path.LINETO
    codes[0] = mpath.Path.MOVETO

    all_codes = np.concatenate((codes, codes))

    # Concatenate the inside and outside subpaths together, changing their order as needed
    outside, inside = (1, -1)
    vertices = np.concatenate((vu[::outside], vi[::inside]))

    # Create the Path object
    path = mpath.Path(vertices, all_codes)
    patch = mpatches.PathPatch(path, **kwargs)

    return patch


def _get_circle_vertices(r):
    t = np.arange(0, np.pi * 2.0, 0.01)
    t = t.reshape((len(t), 1))
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.hstack((x, y))


# ======================================================================================================================
# Add point to Matplotlib 3D surfaces
# ======================================================================================================================
# based on: https://stackoverflow.com/questions/51241367/matplotlib-surface-plot-hides-scatter-points-which-should-be-in-front

def add_point(ax, x, y, z, fc=None, ec=None, radius=0.005):
    xy_len, z_len = ax.get_figure().get_size_inches()

    axis_length = [x[1] - x[0] for x in [ax.get_xbound(), ax.get_ybound(), ax.get_zbound()]]
    axis_rotation = {'z': ((x, y, z), axis_length[1] / axis_length[0]),
                     'y': ((x, z, y), axis_length[2] / axis_length[0] * xy_len / z_len),
                     'x': ((y, z, x), axis_length[2] / axis_length[1] * xy_len / z_len)}

    for a, ((x0, y0, z0), ratio) in axis_rotation.items():
        p = Ellipse((x0, y0), width=radius, height=radius * ratio, fc=fc, ec=ec)
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=z0, zdir=a)
