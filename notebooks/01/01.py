# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # `01.ipynb`

# ## 🐍 Python imports 🐍

# +
import glob
import os.path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# -

# ## 📁 Data processing 📁
#
# ---
#
# ### ⚪ Import all excel sheets at once
#
# https://stackoverflow.com/a/21232849
#
# https://stackoverflow.com/a/918178
#
# ### ⚫ Get rid of "no defaults" error
#
# <center>
#     <img src="images/error_read_excel.png" style="width:80%;">
# </center>
#
# https://stackoverflow.com/questions/66214951/how-to-deal-with-warning-workbook-contains-no-default-style-apply-openpyxls

# +
files = glob.glob("data/*.xlsx")
dfs = {}

for file in files:
    file_name = os.path.splitext(file)[0].replace("data/", "")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        df = pd.read_excel(file)
    dfs[file_name] = df
# -

# ## 🌸 Playground 🌸

type(dfs)

dfs.keys()

type(dfs["Teams"])

dfs["Teams"]

# ## ❄ Plots ❄


