"""Plain matplotlib style for the lane's PNG figures (reference palette of the dataviz skill)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
SEQ = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#e4e3df'


def seq(k, K):
    """k-th of K ordinal steps of the blue ramp (light -> dark, never lighter than step 250)."""
    if K == 1:
        return SEQ[5]
    return SEQ[round(k * (len(SEQ) - 1) / (K - 1))]


def setup():
    plt.rcParams.update({
        'figure.dpi': 110, 'savefig.dpi': 150, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
        'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
        'axes.spines.right': False, 'lines.linewidth': 1.6, 'legend.frameon': False, 'font.size': 9.5,
        'axes.titlesize': 10, 'axes.titleweight': 'normal'})
