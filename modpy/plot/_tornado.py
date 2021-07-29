import numpy as np
from matplotlib import pyplot as plt

# ###############################################################################
# # The data (change all of this to your actual data, this is just a mockup)
# variables = [
#     'apple',
#     'juice',
#     'orange',
#     'peach',
#     'gum',
#     'stones',
#     'bags',
#     'lamps',
# ]
#
# base = 3000
#
# lows = np.array([
#     base - 246 / 2,
#     base - 1633 / 2,
#     base - 500 / 2,
#     base - 150 / 2,
#     base - 35 / 2,
#     base - 36 / 2,
#     base - 43 / 2,
#     base - 37 / 2,
# ])
#
# values = np.array([
#     246,
#     1633,
#     500,
#     150,
#     35,
#     36,
#     43,
#     37,
# ])
#
# ###############################################################################
# # The actual drawing part
#
# # The y position for each variable
# ys = range(len(values))[::-1]  # top to bottom
#
# # Plot the bars, one by one
# for y, low, value in zip(ys, lows, values):
#     # The width of the 'low' and 'high' pieces
#     low_width = base - low
#     high_width = low + value - base
#
#     # Each bar is a "broken" horizontal bar chart
#     plt.broken_barh(
#         [(low, low_width), (base, high_width)],
#         (y - 0.4, 0.8),
#         facecolors=['white', 'white'],  # Try different colors if you like
#         edgecolors=['black', 'black'],
#         linewidth=1,
#     )
#
#     # Display the value as text. It should be positioned in the center of
#     # the 'high' bar, except if there isn't any room there, then it should be
#     # next to bar instead.
#     x = base + high_width / 2
#     if x <= base + 50:
#         x = base + high_width + 50
#     plt.text(x, y, str(value), va='center', ha='center')
#
# # Draw a vertical line down the middle
# plt.axvline(base, color='black')
#
# # Position the x-axis on the top, hide all the other spines (=axis lines)
# axes = plt.gca()  # (gca = get current axes)
# axes.spines['left'].set_visible(False)
# axes.spines['right'].set_visible(False)
# axes.spines['bottom'].set_visible(False)
# axes.xaxis.set_ticks_position('top')
#
# # Make the y-axis display the variables
# plt.yticks(ys, variables)
#
# # Set the portion of the x- and y-axes to show
# plt.xlim(base - 1000, base + 1000)
# plt.ylim(-1, len(variables))
#
# plt.show()

col_low = np.array([253., 191., 111.]) / 255.
col_high = np.array([32., 120., 180.]) / 255.


def tornado(ax, low, high, base=0., labels=(), facecolors=(col_low, col_high)):
    """
    Draws a tornado chart.

    Originally based on:
    https://stackoverflow.com/questions/32132773/a-tornado-chart-and-p10-p90-in-python-matplotlib

    Parameters
    ----------
    ax : matplotlib.Axes
        Axes on which to draw tornado chart.
    low : array_like, shape (n,)
        Values of low case results.
    high : array_like, shae (n,)
        Values of high case results.
    base : float
        Base case value.
    labels : tuple
        Labels for the y-axis.
    facecolors : tuple
        Tuple of (color_low, color_high).
    """

    # ensure consistent input ------------------------------------------------------------------------------------------
    low = np.array(low)
    high = np.array(high)

    n = low.size

    if high.size != n:
        raise ValueError('`low` ({}) and `high` ({}) must have the same length.'.format(n, high.size))

    if not labels:
        labels = [str(i) for i in range(1, n + 1)]

    if len(labels) != n:
        raise ValueError('`labels` ({}) must have the same length as `low` and `high` ({}).'.format(len(labels), n))

    if np.any(low > base):
        raise ValueError('All values of `low` must be less than or equal to `base`.')

    if np.any(high < base):
        raise ValueError('All values of `high` must be greater than or equal to `base`.')

    # sort according to largest difference -----------------------------------------------------------------------------
    diff = high - low
    idx = np.argsort(diff)[::-1]
    low = low[idx]
    high = high[idx]
    labels = [labels[i] for i in idx]

    # for labeling
    min_dist = np.amax(diff) * 0.05

    # draw chart -------------------------------------------------------------------------------------------------------
    # The y position for each variable
    ys = range(n)[::-1]  # top to bottom

    # Plot the bars, one by one
    for y, l, h in zip(ys, low, high):
        # The width of the 'low' and 'high' pieces
        low_width = base - l
        high_width = h - base

        # Each bar is a "broken" horizontal bar chart
        ax.broken_barh(
            [(l, low_width), (base, high_width)],
            (y - 0.4, 0.8),
            facecolors=facecolors,
            edgecolors=['black', 'black'],
            linewidth=1,
        )

        # display text for negative increments
        xl = base - low_width / 2.
        if xl >= base - min_dist:
            xl = base - low_width - min_dist
            ha = 'right'
        else:
            ha = 'center'

        low_width = int(low_width) if low_width >= 10. else low_width
        ax.text(xl, y, str(low_width), va='center', ha=ha)

        # display text for positive increments
        xh = base + high_width / 2.
        if xh <= base + min_dist:
            xh = base + high_width + min_dist
            ha = 'left'
        else:
            ha = 'center'

        high_width = int(high_width) if high_width >= 10 else high_width
        ax.text(xh, y, str(high_width), va='center', ha=ha)

    # Draw a vertical line down the middle
    ax.axvline(base, color='black')

    # Position the x-axis on the top, hide all the other spines (=axis lines)
    #ax.spines['left'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_ticks_position('top')

    # Make the y-axis display the variables
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.tick_params(axis='y', which='both', length=0)

    # set grid
    ax.grid(True)
    ax.set_axisbelow(True)

    # Set the portion of the x- and y-axes to show
    ax.set_ylim([-1, n])
