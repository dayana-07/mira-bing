#!/usr/bin/env python
# coding: utf-8

# In[49]:


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

# In[50]:


Teams = pd.read_excel('Documents\mira-bing\data\Teams.xlsx')
gender = pd.read_excel('Documents\mira-bing\data\EntriesGender.xlsx')


# In[51]:


Teams.head()


# In[52]:


gender.head()


# In[53]:


Teams.describe()


# In[54]:


gender.describe()


# ## Merging the DataFrames

# In[55]:


TbyG= Teams.merge(gender, on = 'Discipline')
TbyG


# In[56]:


TbyG.describe()


# In[57]:


TbyG.head(10)


# ## But Wait! There's another way to inspect- Introducing Pandas_profiling

# In[58]:


help(pandas_profiling) #https://github.com/ydataai/pandas-profiling/blob/develop/README.md


# In[59]:


TbyG.profile_report()


# # Looking at Olympic Medals

# In[60]:


Medals = pd.read_csv("Documents\mira-bing\data\Medals.csv")


# In[81]:


hist_med = pd.read_csv("Documents\mira-bing\data\Summer_medals_History.csv")


# In[62]:


Medals.head()


# In[63]:


hist_med.head()


# #### Align Order of Columns

# In[64]:


cols = Medals.columns.tolist()
cols


# In[65]:


Medals = Medals[['Team/NOC', 'Total', 'Gold', 'Silver', 'Bronze', 'Rank', 'Rank by Total']]
Medals.head()


# ## pair-wise plot of Medals by Country

# In[66]:


sns.pairplot(Medals, hue="Team/NOC", height = 2.5)


# ## Hypothesis:The number of medals won in a previous Olympics can predict the country most likely to have won the medals

# ### examining historical medal data

# In[67]:


X = hist_med.drop(columns = 'Country')
y = hist_med['Country']
model = DecisionTreeClassifier()
model.fit(X, y)

# testing theory
predictions = model.predict([ [2002, 572, 780, 650], [4, 3, 0, 1] ])
predictions


# #### so we can do it, but is it accurate?

# In[68]:


from sklearn.metrics import accuracy_score
X = hist_med.drop(columns =['Country'])
y = hist_med['Country']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# #### Checking the accuracy at 75,25

# In[69]:


model2 = DecisionTreeClassifier()
model2.fit(X_train, y_train)

predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)
score


# Note: The Historical Medal Data set is the accrued medal count over the 26 years the Summer Olympics had been held when the database was assembled. This gives us two options for finding a useful model.
# 1- Finding the average of the medals won over the 26 years included or
# 2- Multiplying the medals won in the most recent year by 26 as an estimate of how many medals are expected over 26 years.
# We examined both options to evaluate performance of the models.

# #### avg medals per country 

# In[70]:


def divide_by_26_games(number): 
    return number/26
 
# executing the function
hist_med_avg  = hist_med[['Total', 'Golds','Silvers', 'Bronzes']].apply(divide_by_26_games)

hist_med_avg


# #### multiply current medals by 26

# In[71]:


def multiply_by_26_games(number):
    return number *26
#executing the function
Medals_Multiplied = Medals[['Total', 'Gold', 'Silver', 'Bronze']].apply(multiply_by_26_games)
Medals_Multiplied


# ## Can the number of medals won in an Olympics predict the country most likely to have won the medals?

# In[72]:


X_Medals = Medals_Multiplied 
y_Medals = Medals['Team/NOC']
model_Medals_Multiplied = DecisionTreeClassifier()
model_Medals_Multiplied.fit(X_Medals, y_Medals)

# testing theory
predictions = model_Medals_Multiplied.predict([ [77,22,30,25], [4, 3, 0, 1] ])
predictions


# In[73]:


X = hist_med_avg
y = hist_med['Country']
model = DecisionTreeClassifier()
model.fit(X, y)

# testing theory
predictions = model.predict([ [77, 22, 30, 25], [4, 3, 0, 1] ])
predictions


# ## so we can do it, but is it accurate?

# ## Checking the accuracy at 75,25

# In[74]:


from sklearn.metrics import accuracy_score 

# Training the Medals_Multiplied set
X_Medals_train, X_Medals_test, y_Medals_train, y_Medals_test = train_test_split(X_Medals, y_Medals, test_size = 0.25)

model_Medals_Multiplied.fit(X_Medals_train, y_Medals_train)

predictions = model_Medals_Multiplied.predict(X_Medals_test)

# examing the accuracy score
score = accuracy_score(y_Medals_test, predictions)
score


# In[75]:


# Training the Historical Medals averaged by years
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
model2 = DecisionTreeClassifier()
model2.fit(X_train, y_train)

predictions = model.predict(X_test)

# examing the accuracy score
score = accuracy_score(y_test, predictions)
score


# ### Accuracy is questionable for both models, but let's investigate what happens with the entire new Medals list as the prediction data

# In[76]:


List_medals = Medals_Multiplied.values.tolist()
List_medals


# ### Multiplied Medals prediction

# In[77]:


model_Medals_Multiplied = DecisionTreeClassifier()
model_Medals_Multiplied.fit(X_Medals_train, y_Medals_train)

# testing theory
predictions = model_Medals_Multiplied.predict(List_medals)
predictions


# ### Averaged Historical Medals 

# In[78]:


model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# testing theory
predictions = model.predict(List_medals)
predictions


# ### Finally, let's see what it looks like in panda_profiler

# In[79]:


Medals.profile_report()


# In[80]:


hist_med.profile_report()

