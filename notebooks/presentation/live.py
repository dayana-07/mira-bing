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

# # `presentation.ipynb`
#
# <center>
#     <img src="../../images/presentation.png" style="width:60%; border-radius:2%">
# </center>
#
# [🖼 Shinnosuke Ando | Unsplash](https://unsplash.com/photos/VUsicdRLMFQ)
#
# ---

# ## ❄ Ai ❄

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
data = "../../data/"
path = data
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

plt.style.use('dark_background')

# +
df = dfs["EntriesGender"]
# df.columns

pan = {"Female": "#ff218e", "Total": "#fcd800", "Male": "#0194fc"}
# bi = {"Female": "#F7A8B8", "Total": "#8C4799", "Male": "#55CDFC"}
# trans = {"Female": "#55CDFC", "Total": "#FFFFFF", "Male": "#F7A8B8"}
# -

df = df.set_index(df["Discipline"]).drop(columns="Discipline")

plt.rcParams['figure.figsize'] = [20, 10]

df[["Female", "Total", "Male"]].plot.bar(title="Gender by discipline", color = pan)

# ## 📚 References 📚

# ## 🌈 Color schemes 🌈 
#
# - [🔗 Bisexuality Flag Colors Hex, RGB & CMYK Codes | schemecolor.com](https://www.schemecolor.com/bisexuality-flag-colors.php)
# - [🔗 Pansexuality Flag Colors Hex, RGB & CMYK Codes | schemecolor.com](https://www.schemecolor.com/pansexuality-flag-colors.php)
# - [🔗 Transgender Pride Flag Colors Hex, RGB & CMYK Codes | schemecolor.com](https://www.schemecolor.com/transgender-pride-flag-colors.php)

# ### ❄❄ Data scraping
#
# [🔗 athena | GitHub](https://github.com/Ai-Yukino/athena)

# ### ❄❄ Summary

# ## ❄ Cherylyn ❄ 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_profiling
import seaborn as sns
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib as jb

# ## The usual way we inspect data

# +
# Teams = pd.read_excel('Documents\mira-bing\data\Teams.xlsx')
# gender = pd.read_excel('Documents\mira-bing\data\EntriesGender.xlsx')

Teams = pd.read_excel(data + 'Teams.xlsx')
gender = pd.read_excel(data + 'EntriesGender.xlsx')
# -

Teams.head()

gender.head()

Teams.describe()

gender.describe()

# ## Merging the DataFrames

TbyG= Teams.merge(gender, on = 'Discipline')
TbyG

TbyG.describe()

TbyG.head(10)

# ## But Wait! There's another way to inspect- Introducing Pandas_profiling

help(pandas_profiling) #https://github.com/ydataai/pandas-profiling/blob/develop/README.md

TbyG.profile_report()

# # Looking at Olympic Medals

# Medals = pd.read_csv("Documents\mira-bing\data\Medals.csv")
Medals = pd.read_csv(data + "Medals.csv")

# hist_med = pd.read_csv("Documents\mira-bing\data\Summer_medals_History.csv")
hist_med = pd.read_csv(data + "Summer_medals_History.csv")

Medals.head()

hist_med.head()

# #### Align Order of Columns

cols = Medals.columns.tolist()
cols

Medals = Medals[['Team/NOC', 'Total', 'Gold', 'Silver', 'Bronze', 'Rank', 'Rank by Total']]
Medals.head()

# ## pair-wise plot of Medals by Country

sns.pairplot(Medals, hue="Team/NOC", height = 2.5)

# ## Hypothesis:The number of medals won in a previous Olympics can predict the country most likely to have won the medals

# ### examining historical medal data

# +
X = hist_med.drop(columns = 'Country')
y = hist_med['Country']
model = DecisionTreeClassifier()
model.fit(X, y)

# testing theory
predictions = model.predict([ [2002, 572, 780, 650], [4, 3, 0, 1] ])
predictions
# -

# #### so we can do it, but is it accurate?

from sklearn.metrics import accuracy_score
X = hist_med.drop(columns =['Country'])
y = hist_med['Country']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# #### Checking the accuracy at 75,25

# +
model2 = DecisionTreeClassifier()
model2.fit(X_train, y_train)

predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)
score


# -

# Note: The Historical Medal Data set is the accrued medal count over the 26 years the Summer Olympics had been held when the database was assembled. This gives us two options for finding a useful model.
# 1- Finding the average of the medals won over the 26 years included or
# 2- Multiplying the medals won in the most recent year by 26 as an estimate of how many medals are expected over 26 years.
# We examined both options to evaluate performance of the models.

# #### avg medals per country 

# +
def divide_by_26_games(number): 
    return number/26
 
# executing the function
hist_med_avg  = hist_med[['Total', 'Golds','Silvers', 'Bronzes']].apply(divide_by_26_games)

hist_med_avg


# -

# #### multiply current medals by 26

def multiply_by_26_games(number):
    return number *26
#executing the function
Medals_Multiplied = Medals[['Total', 'Gold', 'Silver', 'Bronze']].apply(multiply_by_26_games)
Medals_Multiplied

# ## Can the number of medals won in an Olympics predict the country most likely to have won the medals?

# +
X_Medals = Medals_Multiplied 
y_Medals = Medals['Team/NOC']
model_Medals_Multiplied = DecisionTreeClassifier()
model_Medals_Multiplied.fit(X_Medals, y_Medals)

# testing theory
predictions = model_Medals_Multiplied.predict([ [77,22,30,25], [4, 3, 0, 1] ])
predictions

# +
X = hist_med_avg
y = hist_med['Country']
model = DecisionTreeClassifier()
model.fit(X, y)

# testing theory
predictions = model.predict([ [77, 22, 30, 25], [4, 3, 0, 1] ])
predictions
# -

# ## so we can do it, but is it accurate?

# ## Checking the accuracy at 75,25

# +
from sklearn.metrics import accuracy_score 

# Training the Medals_Multiplied set
X_Medals_train, X_Medals_test, y_Medals_train, y_Medals_test = train_test_split(X_Medals, y_Medals, test_size = 0.25)

model_Medals_Multiplied.fit(X_Medals_train, y_Medals_train)

predictions = model_Medals_Multiplied.predict(X_Medals_test)

# examing the accuracy score
score = accuracy_score(y_Medals_test, predictions)
score

# +
# Training the Historical Medals averaged by years
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
model2 = DecisionTreeClassifier()
model2.fit(X_train, y_train)

predictions = model.predict(X_test)

# examing the accuracy score
score = accuracy_score(y_test, predictions)
score

# -

# ### Accuracy is questionable for both models, but let's investigate what happens with the entire new Medals list as the prediction data

List_medals = Medals_Multiplied.values.tolist()
List_medals

# ### Multiplied Medals prediction

# +
model_Medals_Multiplied = DecisionTreeClassifier()
model_Medals_Multiplied.fit(X_Medals_train, y_Medals_train)

# testing theory
predictions = model_Medals_Multiplied.predict(List_medals)
predictions
# -

# ### Averaged Historical Medals 

# +
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# testing theory
predictions = model.predict(List_medals)
predictions
# -

# ## ❄ Dayana ❄ 

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
path = "../../data/"
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

type(dfs["Medals"])

Med = dfs["Medals"]
Med

Med.describe()

Gend = dfs["EntriesGender"]
Gend

Gend.describe()

Aths = dfs["Athletes"]
Aths

Aths.describe()

Teams = dfs['Teams']
Teams

Teams.describe()

Coach = dfs['Coaches']
Coach

Coach.describe()

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
