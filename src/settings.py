# settings.py
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import plots

carpediem_dir = '../data/carpediem'
curated_dir = '../data/curated'
mimic_dir = '../data/mimiciv_included'
calc_dir = '../data/calculated'
plots_dir = '../plots'

mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.reload_library()
rcparams = plots.stdrcparams()
mpl.rcParams.update(rcparams)
