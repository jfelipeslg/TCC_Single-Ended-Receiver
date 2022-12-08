# Spyder IDE

import pickle
from IPython import get_ipython


list_auxM = [4, 16, 64]
list_auxLtotal = [0, 160]
list_auxLOSPR = [value for value in range(6,18,2)]


for auxLtotal in list_auxLtotal:
    for auxM in list_auxM:
        for auxLOSPR in list_auxLOSPR:
            get_ipython().run_line_magic('run', './main_Fig3A.ipynb')