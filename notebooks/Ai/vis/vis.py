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

# # `vis.ipynb`

# ## 🐍 Python imports 🐍

# ### 🗃 File processing

import glob
import os.path

# ### ⚠ Warning messages

import warnings

# ### 🎨 Visualization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ## 📁 Data processing 📁
#
# ---
#
# <details>
#     <summary> 🔗 Show links</summary>
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
#
# </details>

# ### 🍃 Extract
#
# ---
#
# Store each excel file as a DataFrame in a dictionary `dfs`

# +
path = "../../../data/"
# files = glob.glob("../../data/*.xlsx")
files = glob.glob(path + "*.xlsx")
dfs = {}

for file in files:
    file_name = os.path.splitext(file)[0].replace(path, "")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        df = pd.read_excel(file)
    dfs[file_name] = df
# -

# ### 💧🔥 Transform

# ### 🍵 Load

# #### 📝 Example: How to get a DataFrame from the `dfs` dictionary

# +
# type(dfs)
# dfs.keys()
# type(dfs["Athletes"])
# dfs["Athletes"]["Name"]
# -

# ## ❄ Visualizations ❄
#

# ### ❄❄ Gender

plt.rcParams['figure.figsize'] = [20, 10]

plt.style.use('dark_background')

# +
df = dfs["EntriesGender"]
# df.columns

pan = {"Female": "#ff218e", "Total": "#fcd800", "Male": "#0194fc"}
# bi = {"Female": "#F7A8B8", "Total": "#8C4799", "Male": "#55CDFC"}
# trans = {"Female": "#55CDFC", "Total": "#FFFFFF", "Male": "#F7A8B8"}
# -

df = df.set_index(df["Discipline"]).drop(columns="Discipline")

df[["Female", "Total", "Male"]].plot.bar(title="Gender by discipline", color = pan)

# ## 📚 References 📚

# ## 🌈 Color schemes 🌈 
#
# - [🔗 Bisexuality Flag Colors Hex, RGB & CMYK Codes | schemecolor.com](https://www.schemecolor.com/bisexuality-flag-colors.php)
# - [🔗 Pansexuality Flag Colors Hex, RGB & CMYK Codes | schemecolor.com](https://www.schemecolor.com/pansexuality-flag-colors.php)
# - [🔗 Transgender Pride Flag Colors Hex, RGB & CMYK Codes | schemecolor.com](https://www.schemecolor.com/transgender-pride-flag-colors.php)
