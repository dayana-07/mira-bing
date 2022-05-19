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

# # `template.ipynb`

# ## 🐍 Python imports 🐍

# ### 🗃 File processing

import glob
import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ### ⚠ Warning messages

import warnings

# ### 🎨 Visualization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

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

# + [markdown] tags=[]
# ### 💧🔥 Transform
# -

# ### 🍵 Load

# #### 📝 Example: How to get a DataFrame from the `dfs` dictionary

type(dfs)

dfs.keys()

type(dfs["../../../data\\Medals"])

Med = dfs["../../../data\\Medals"]
Med

Med.describe()

Gend = dfs["../../../data\\EntriesGender"]
Gend

Gend.describe()

Aths = dfs["../../../data\\Athletes"]
Aths

Aths.describe()

Teams = dfs['../../../data\\Teams']
Teams

Teams.describe()

Coach = dfs['../../../data\\Coaches']
Coach

Coach.describe()

#

# ## ❄ Visualizations ❄
#
# ---
#
# TBA

# +
Gend.plot(x="Discipline", y=["Female","Male"], kind="bar", figsize=(15,5))
 
# print bar graph
plt.show()


# +
Med.plot(x="Team/NOC", y=["Gold", "Silver","Bronze"], kind="bar", figsize=(20,5))
 
# print bar graph
plt.show()
# -

