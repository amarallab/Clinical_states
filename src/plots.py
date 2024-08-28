# figures
import matplotlib.mlab as ml
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.ticker as plticker
import palettable
import seaborn as sns

def stdfigsize(scale=1, nx=1, ny=1, ratio=1.3):
    """
    Returns a tuple to be used as figure size.
    -------
    returns (7*ratio*scale*nx, 7.*scale*ny)
    By default: ratio=1.3
    If ratio<0 them ratio = golden ratio
    """
    if ratio < 0:
        ratio = 1.61803398875
    return((7*ratio*scale*nx, 7*scale*ny))

def stdrcparams(usetex=False):
    """
    Set several mpl.rcParams and sns.set_style for my taste.
    ----
    usetex = True
    ----
    """
    sns.set_style("white")
    sns.set_style({"xtick.direction": "in",
                 "ytick.direction": "in"})
    rcparams = {
#     'text.usetex': usetex,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Tahoma', 'DejaVu Sans', 'Lucida Grande', 'Verdana'],
        'axes.labelsize': 28,
        'axes.titlesize': 28,
        'legend.fontsize': 20,
        'ytick.right': 'off',
        'xtick.top': 'off',
        'ytick.left': 'on',
        'xtick.bottom': 'on',
        'xtick.labelsize': '25',
        'ytick.labelsize': '25',
        'axes.linewidth': 2.5,
        'xtick.major.width': 1.8,
        'xtick.minor.width': 1.8,
        'xtick.major.size': 14,
        'xtick.minor.size': 7,
        'xtick.major.pad': 10,
        'xtick.minor.pad': 10,
        'ytick.major.width': 1.8,
        'ytick.minor.width': 1.8,
        'ytick.major.size': 14,
        'ytick.minor.size': 7,
        'ytick.major.pad': 10,
        'ytick.minor.pad': 10,
        'axes.labelpad': 15,
        'axes.titlepad': 15,
        'axes.spines.right': False,
        'axes.spines.top': False}
    return rcparams

    # mpl.rcParams.update(rcparams) 
    # mpl.rcParams['lines.linewidth'] = 5
    # mpl.rcParams['pdf.fonttype'] = 42 

# stdrcparams(usetex=True)
# figsize=stdfigsize(ratio=-1)
# xs,ys=figsize


def stdrcparams1():
    
    sns.set_style("white")
    
    rcparams = {
        'pdf.fonttype' : 42,
        'ps.fonttype' : 42,
        'font.family': 'serif',
        'font.sans-serif': ['Times New Roman'],
        'font.size': 18,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'legend.fontsize': 14,
        'ytick.right': 'off',
        'xtick.top': 'off',
        'ytick.left': 'on',
        'xtick.bottom': 'on',
        'xtick.labelsize': '18',
        'ytick.labelsize': '18',
        'axes.linewidth': 2,
        'xtick.major.width': 1.2,
        'xtick.minor.width': 1.2,
        'xtick.major.size': 10,
        'xtick.minor.size': 5,
        'xtick.major.pad': 10,
        'xtick.minor.pad': 10,
        'ytick.major.width': 1.2,
        'ytick.minor.width': 1.2,
        'ytick.major.size': 10,
        'ytick.minor.size': 5,
        'ytick.major.pad': 10,
        'ytick.minor.pad': 10,
        'axes.labelpad': 15,
        'axes.titlepad': 15,
        'axes.spines.right': True,
        'axes.spines.top': True}
    return rcparams