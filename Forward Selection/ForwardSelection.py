#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('/Users/fionafei/Desktop/Chicago/WIN22/Linear and Non-Linear Models/Assignment2/Sample Code')
import Regression
from scipy.special import loggamma
from scipy.stats import norm
from scipy.stats import chi2


# In[2]:


df = pd.read_csv("claim_history.csv")


# In[3]:


trainData = df[['KIDSDRIV', 'HOMEKIDS', 'TRAVTIME', 'MSTATUS', 'TIF', 'CAR_TYPE', 
         'REVOKED', 'MVR_PTS', 'CAR_AGE', 'URBANICITY', 'CLM_COUNT', 'EXPOSURE']].dropna()


# In[4]:


y = trainData['CLM_COUNT']
x = trainData['EXPOSURE']
logX = np.log(trainData['EXPOSURE'])


# In[5]:


trainData.head()


# In[6]:


# Set some options for printing all the columns
np.set_printoptions(precision = 10, threshold = sys.maxsize)
np.set_printoptions(linewidth = np.inf)

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)
pd.set_option('precision', 10)

pd.options.display.float_format = '{:,.7e}'.format


# ## Question 1 (20 points)

# ### a)	(20 points) For each predictor, generate a scatterplot chart that shows the number of claims by the predictor’s values.  Also, color-code the markers by the exposure values.  Please display the predictor’s values are displayed in ascending lexical order.

# In[7]:


# KIDSDRIV
y = trainData['CLM_COUNT']
x = trainData['KIDSDRIV'].astype('category')
plt.figure(dpi = 100)
scatter = plt.scatter(x, y, c = trainData.EXPOSURE)
plt.xlabel('KIDSDRIV')
plt.ylabel('CLM_COUNT')
plt.xticks(range(5))
plt.yticks(range(10))
plt.grid(axis = 'both')
cbar = plt.colorbar(scatter)
cbar.set_label('EXPOSURE')
plt.show()


# In[8]:


# TRAVTIME
y = trainData['CLM_COUNT']
x = trainData['TRAVTIME'].astype('category')
plt.figure(dpi = 100)
scatter = plt.scatter(x, y, c = trainData.EXPOSURE)
plt.xlabel('TRAVTIME')
plt.ylabel('CLM_COUNT')
plt.xticks()
plt.yticks(range(10))
plt.grid(axis = 'both')
cbar = plt.colorbar(scatter)
cbar.set_label('EXPOSURE')
plt.show()


# In[9]:


# MSTATUS
y = trainData['CLM_COUNT']
x = trainData['MSTATUS'].astype('category')
plt.figure(dpi = 100)
scatter = plt.scatter(x, y, c = trainData.EXPOSURE)
plt.xlabel('MSTATUS')
plt.ylabel('CLM_COUNT')
plt.xticks()
plt.yticks()
plt.grid(axis = 'both')
cbar = plt.colorbar(scatter)
cbar.set_label('EXPOSURE')
plt.show()


# In[10]:


# TIF
y = trainData['CLM_COUNT']
x = trainData['TIF'].astype('category')
plt.figure(dpi = 100)
scatter = plt.scatter(x, y, c = trainData.EXPOSURE)
plt.xlabel('TIF')
plt.ylabel('CLM_COUNT')
plt.xticks()
plt.yticks()
plt.grid(axis = 'both')
cbar = plt.colorbar(scatter)
cbar.set_label('EXPOSURE')
plt.show()


# In[11]:


# CAR_TYPE
y = trainData['CLM_COUNT']
x = trainData['CAR_TYPE'].astype('category')
plt.figure(dpi = 100)
scatter = plt.scatter(x, y, c = trainData.EXPOSURE)
plt.xlabel('CAR_TYPE')
plt.ylabel('CLM_COUNT')
plt.xticks()
plt.yticks()
plt.grid(axis = 'both')
cbar = plt.colorbar(scatter)
cbar.set_label('EXPOSURE')
plt.show()


# In[12]:


# REVOKED
y = trainData['CLM_COUNT']
x = trainData['REVOKED'].astype('category')
plt.figure(dpi = 100)
scatter = plt.scatter(x, y, c = trainData.EXPOSURE)
plt.xlabel('REVOKED')
plt.ylabel('CLM_COUNT')
plt.xticks()
plt.yticks()
plt.grid(axis = 'both')
cbar = plt.colorbar(scatter)
cbar.set_label('EXPOSURE')
plt.show()


# In[13]:


# MVR_PTS
y = trainData['CLM_COUNT']
x = trainData['MVR_PTS'].astype('category')
plt.figure(dpi = 100)
scatter = plt.scatter(x, y, c = trainData.EXPOSURE)
plt.xlabel('MVR_PTS')
plt.ylabel('CLM_COUNT')
plt.xticks()
plt.yticks()
plt.grid(axis = 'both')
cbar = plt.colorbar(scatter)
cbar.set_label('EXPOSURE')
plt.show()


# In[14]:


#CAR_AGE
y = trainData['CLM_COUNT']
x = trainData['CAR_AGE'].astype('category')
plt.figure(dpi = 100)
scatter = plt.scatter(x, y, c = trainData.EXPOSURE)
plt.xlabel('CAR_AGE')
plt.ylabel('CLM_COUNT')
plt.xticks()
plt.yticks()
plt.grid(axis = 'both')
cbar = plt.colorbar(scatter)
cbar.set_label('EXPOSURE')
plt.show()


# In[15]:


# URBANICITY
y = trainData['CLM_COUNT']
x = trainData['URBANICITY'].astype('category')
plt.figure(dpi = 100)
scatter = plt.scatter(x, y, c = trainData.EXPOSURE)
plt.xlabel('URBANICITY')
plt.ylabel('CLM_COUNT')
plt.xticks()
plt.yticks()
plt.grid(axis = 'both')
cbar = plt.colorbar(scatter)
cbar.set_label('EXPOSURE')
plt.show()


# ## Question 2 (40 points)

# ### Enter the predictors into your model using Forward Selection.  The Entry Threshold is 0.05.
# ### a)	(15 points).  Please provide a summary report of the Forward Selection. The report should include (1) the step number, (2) the predictor entered, (3) the number of non-aliased parameters in the current model, (4) the log-likelihood value of the current model, (5) the Deviance Chi-squares statistic between the current and the previous models, (6) the corresponding Deviance Degree of Freedom, and (7) the corresponding Chi-square significance.
# 

# In[16]:


# Intercept only model

X_train = trainData[['CLM_COUNT']].copy()
X_train.insert(0, 'Intercept', 1.0)
X_train.drop(columns = ['CLM_COUNT'], inplace = True)

y_train = trainData['CLM_COUNT']
e_train = trainData['EXPOSURE']
o_train = np.log(trainData['EXPOSURE'])

step_summary = pd.DataFrame()

outList = Regression.PoissonModel(X_train, y_train, o_train)
llk_0 = outList[3]
df_0 = len(outList[4])
step_summary = step_summary.append([['Intercept', df_0, llk_0, np.nan, np.nan, np.nan]], ignore_index = True)


# In[17]:


# Find the first predictor
step_detail = pd.DataFrame()

car_type = trainData[['CAR_TYPE']].astype('category')
term_car_type = pd.get_dummies(car_type)

mstatus = trainData[['MSTATUS']].astype('category')
term_mstatus = pd.get_dummies(mstatus, drop_first=True)

revoked = trainData[['REVOKED']].astype('category')
term_revoked = pd.get_dummies(revoked, drop_first=True)

urban = trainData[['URBANICITY']].astype('category')
term_urban = pd.get_dummies(urban, drop_first=True)


# In[18]:


# Find the first predictor
step_detail = pd.DataFrame()

# Try Intercept + KIDSDRIV
X = X_train.join(trainData[['KIDSDRIV']])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ KIDSDRIV', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)
    
# Try Intercept + HOMEKIDS
X = X_train.join(trainData[["HOMEKIDS"]])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ HOMEKIDS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Try Intercept + TRAVTIME
X = X_train.join(trainData[['TRAVTIME']])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TRAVTIME', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Try Intercept + MSTATUS
X = X_train.join(term_mstatus)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ MSTATUS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Try Intercept + TIF
X = X_train.join(trainData[['TIF']])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TIF', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)


# Try Intercept + CAR_TYPE
X = X_train.join(term_car_type)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ CAR_TYPE', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Try Intercept + REVOKED
X = X_train.join(term_revoked)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ REVOKED', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Try Intercept + MVR_PTS
X = X_train.join(trainData[['MVR_PTS']]) 
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ MVR_PTS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)


# Try Intercept + CAR_AGE
X = X_train.join(trainData[["CAR_AGE"]])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ CAR_AGE', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)



# Try Intercept + URBAN
X = X_train.join(term_urban)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ URBANICITY', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)


# In[19]:


# Find the first predictor
step_detail


# In[20]:


# Based on the step_detail table above, Intercept + URBANICITY has the lowest deviance significance value. 
# We then update the current model to Intercept + URBANICITY

row = step_detail[step_detail[0] == '+ URBANICITY']
llk_0 = row.iloc[0][2]
df_0 = row.iloc[0][1]
step_summary = step_summary.append(row, ignore_index = True)
X_train = X_train.join(term_urban)


# In[21]:


# Find the second predictor
step_detail = pd.DataFrame()


# Try Intercept + URBANICITY + KIDSDRIV
X = X_train.join(trainData[['KIDSDRIV']])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ KIDSDRIV', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)
    
# Try Intercept + URBANICITY + HOMEKIDS
X = X_train.join(trainData[["HOMEKIDS"]])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ HOMEKIDS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Try Intercept + URBANICITY + TRAVTIME
X = X_train.join(trainData[['TRAVTIME']])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TRAVTIME', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Try Intercept + URBANICITY + MSTATUS
X = X_train.join(term_mstatus)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ MSTATUS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Try Intercept + URBANICITY + TIF
X = X_train.join(trainData[['TIF']])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TIF', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)


# Try Intercept + URBANICITY + CAR_TYPE
X = X_train.join(term_car_type)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ CAR_TYPE', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Try Intercept + URBANICITY + REVOKED
X = X_train.join(term_revoked)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ REVOKED', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Try Intercept + URBANICITY + MVR_PTS
X = X_train.join(trainData[['MVR_PTS']]) 
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ MVR_PTS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)


# Try Intercept + URBANICITY + CAR_AGE
X = X_train.join(trainData[["CAR_AGE"]])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ CAR_AGE', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)


# In[22]:


# Find the second predictor
step_detail


# In[24]:


# Based on the step_detail table above, Intercept + URBANICITY + MVR_PTS has the lowest deviance significance value. 
# We then update the current model to Intercept + URBANICITY + MVR_PTS

row = step_detail[step_detail[0] == '+ MVR_PTS']
llk_0 = row.iloc[0][2]
df_0 = row.iloc[0][1]
step_summary = step_summary.append(row, ignore_index = True)
X_train = X_train.join(trainData[['MVR_PTS']])


# In[25]:


# Find the third predictor
step_detail = pd.DataFrame()


# Try Intercept + URBANICITY + MVR_PTS + KIDSDRIV
X = X_train.join(trainData[['KIDSDRIV']])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ KIDSDRIV', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)
    
# Try Intercept + URBANICITY + MVR_PTS + HOMEKIDS
X = X_train.join(trainData[["HOMEKIDS"]])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ HOMEKIDS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Try Intercept + URBANICITY + MVR_PTS + TRAVTIME
X = X_train.join(trainData[['TRAVTIME']])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TRAVTIME', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Try Intercept + URBANICITY + MVR_PTS + MSTATUS
X = X_train.join(term_mstatus)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ MSTATUS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Try Intercept + URBANICITY + MVR_PTS + TIF
X = X_train.join(trainData[['TIF']])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TIF', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)


# Try Intercept + URBANICITY + MVR_PTS + CAR_TYPE
X = X_train.join(term_car_type)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ CAR_TYPE', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Try Intercept + URBANICITY + MVR_PTS + REVOKED
X = X_train.join(term_revoked)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ REVOKED', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)


# Try Intercept + URBANICITY + MVR_PTS + CAR_AGE
X = X_train.join(trainData[["CAR_AGE"]])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ CAR_AGE', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)


# In[26]:


# Find the third predictor
step_detail


# In[27]:


# Based on the step_detail table above, Intercept + URBANICITY + MVR_PTS + CAR_AGE has the lowest deviance significance value. 
# We then update the current model to Intercept + URBANICITY + MVR_PTS + CAR_AGE

row = step_detail[step_detail[0] == '+ CAR_AGE']
llk_0 = row.iloc[0][2]
df_0 = row.iloc[0][1]
step_summary = step_summary.append(row, ignore_index = True)
X_train = X_train.join(trainData[['CAR_AGE']])


# In[29]:


# Find the fourth predictor
step_detail = pd.DataFrame()


# Try Intercept + URBANICITY + MVR_PTS + CAR_AGE + KIDSDRIV
X = X_train.join(trainData[['KIDSDRIV']])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ KIDSDRIV', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)
    
# Try Intercept + URBANICITY + MVR_PTS + CAR_AGE + HOMEKIDS
X = X_train.join(trainData[["HOMEKIDS"]])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ HOMEKIDS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Try Intercept + URBANICITY + MVR_PTS + CAR_AGE + TRAVTIME
X = X_train.join(trainData[['TRAVTIME']])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TRAVTIME', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Try Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS
X = X_train.join(term_mstatus)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ MSTATUS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Try Intercept + URBANICITY + MVR_PTS + CAR_AGE + TIF
X = X_train.join(trainData[['TIF']])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TIF', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)


# Try Intercept + URBANICITY + MVR_PTS + CAR_AGE + CAR_TYPE
X = X_train.join(term_car_type)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ CAR_TYPE', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Try Intercept + URBANICITY + MVR_PTS + CAR_AGE + REVOKED
X = X_train.join(term_revoked)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ REVOKED', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)


# In[30]:


# Find the fourth predictor
step_detail


# In[31]:


# Based on the step_detail table above, Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS has the lowest deviance significance value. 
# We then update the current model to Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS

row = step_detail[step_detail[0] == '+ MSTATUS']
llk_0 = row.iloc[0][2]
df_0 = row.iloc[0][1]
step_summary = step_summary.append(row, ignore_index = True)
X_train = X_train.join(term_mstatus)


# In[32]:


# Find the fifth predictor
step_detail = pd.DataFrame()


# Try Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + KIDSDRIV
X = X_train.join(trainData[['KIDSDRIV']])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ KIDSDRIV', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)
    
# Try Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + HOMEKIDS
X = X_train.join(trainData[["HOMEKIDS"]])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ HOMEKIDS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Try Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + TRAVTIME
X = X_train.join(trainData[['TRAVTIME']])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TRAVTIME', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)


# Try Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + TIF
X = X_train.join(trainData[['TIF']])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TIF', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)


# Try Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE
X = X_train.join(term_car_type)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ CAR_TYPE', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Try Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + REVOKED
X = X_train.join(term_revoked)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ REVOKED', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)


# In[33]:


# Find the fifth predictor
step_detail


# In[34]:


# Based on the step_detail table above, Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE has the lowest deviance significance value. 
# We then update the current model to Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE 

row = step_detail[step_detail[0] == '+ CAR_TYPE']
llk_0 = row.iloc[0][2]
df_0 = row.iloc[0][1]
step_summary = step_summary.append(row, ignore_index = True)
X_train = X_train.join(term_car_type)

# Find the fifth predictor
step_detail = pd.DataFrame()


# Try Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE + KIDSDRIV
X = X_train.join(trainData[['KIDSDRIV']])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ KIDSDRIV', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)
    
# Try Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE + HOMEKIDS
X = X_train.join(trainData[["HOMEKIDS"]])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ HOMEKIDS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Try Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE + TRAVTIME
X = X_train.join(trainData[['TRAVTIME']])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TRAVTIME', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)


# Try Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE + TIF
X = X_train.join(trainData[['TIF']])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TIF', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)


# Try Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE + REVOKED
X = X_train.join(term_revoked)
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ REVOKED', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)


# In[35]:


# Find the sixth predictor
step_detail


# In[37]:


# Based on the step_detail table above, Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE + REVOKED has the lowest deviance significance value. 
# We then update the current model to Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE + REVOKED

row = step_detail[step_detail[0] == '+ REVOKED']
llk_0 = row.iloc[0][2]
df_0 = row.iloc[0][1]
step_summary = step_summary.append(row, ignore_index = True)
X_train = X_train.join(term_revoked)

# Find the sixth predictor
step_detail = pd.DataFrame()


# Try Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE + REVOKED + KIDSDRIV
X = X_train.join(trainData[['KIDSDRIV']])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ KIDSDRIV', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)
    
# Try Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE + REVOKED + HOMEKIDS
X = X_train.join(trainData[["HOMEKIDS"]])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ HOMEKIDS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Try Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE + REVOKED + TRAVTIME
X = X_train.join(trainData[['TRAVTIME']])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TRAVTIME', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)


# Try Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE + REVOKED + TIF
X = X_train.join(trainData[['TIF']])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TIF', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)


# In[41]:


# Find the seventh(7) predictor
step_detail


# In[43]:


# Based on the step_detail table above, Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE + REVOKED + KIDSDRIV has the lowest deviance significance value. 
# We then update the current model to Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE + REVOKED + KIDSDRIV

row = step_detail[step_detail[0] == '+ KIDSDRIV']
llk_0 = row.iloc[0][2]
df_0 = row.iloc[0][1]
step_summary = step_summary.append(row, ignore_index = True)
X_train = X_train.join(trainData[['KIDSDRIV']])

# Find the seventh(7) predictor
step_detail = pd.DataFrame()

    
# Try Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE + REVOKED + KIDSDRIV + HOMEKIDS
X = X_train.join(trainData[["HOMEKIDS"]])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ HOMEKIDS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)

# Try Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE + REVOKED + KIDSDRIV + TRAVTIME
X = X_train.join(trainData[['TRAVTIME']])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TRAVTIME', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)


# Try Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE + REVOKED + KIDSDRIV + TIF
X = X_train.join(trainData[['TIF']])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TIF', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)


# In[44]:


# Find the 8th predictor
step_detail


# In[45]:


# Based on the step_detail table above, Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE + REVOKED + KIDSDRIV + TRAVTIME has the lowest deviance significance value. 
# We then update the current model to Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE + REVOKED + KIDSDRIV + TRAVTIME

row = step_detail[step_detail[0] == '+ TRAVTIME']
llk_0 = row.iloc[0][2]
df_0 = row.iloc[0][1]
step_summary = step_summary.append(row, ignore_index = True)
X_train = X_train.join(trainData[['TRAVTIME']])

# Find the 8th predictor
step_detail = pd.DataFrame()

    
# Try Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE + REVOKED + KIDSDRIV + TRAVTIME + HOMEKIDS
X = X_train.join(trainData[["HOMEKIDS"]])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ HOMEKIDS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)


# Try Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE + REVOKED + KIDSDRIV + TRAVTIME + TIF
X = X_train.join(trainData[['TIF']])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ TIF', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)


# In[46]:


# Find the 9th predictor
step_detail


# In[47]:


# Based on the step_detail table above, Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE + REVOKED + KIDSDRIV + TRAVTIME + TIF has the lowest deviance significance value. 
# We then update the current model to Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE + REVOKED + KIDSDRIV + TRAVTIME + TIF

row = step_detail[step_detail[0] == '+ TIF']
llk_0 = row.iloc[0][2]
df_0 = row.iloc[0][1]
step_summary = step_summary.append(row, ignore_index = True)
X_train = X_train.join(trainData[['TIF']])

# Find the 9th predictor
step_detail = pd.DataFrame()

    
# Try Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE + REVOKED + KIDSDRIV + TRAVTIME + TIF + HOMEKIDS
X = X_train.join(trainData[["HOMEKIDS"]])
outList = Regression.PoissonModel(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail = step_detail.append([['+ HOMEKIDS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig]], ignore_index = True)


# In[48]:


# Find the 10th predictor
step_detail


# In[49]:


# Based on the step_detail table above, Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE + REVOKED + KIDSDRIV + TRAVTIME + TIF + HOMEKIDS has the lowest deviance significance value. 
# We then update the current model to Intercept + URBANICITY + MVR_PTS + CAR_AGE + MSTATUS + CAR_TYPE + REVOKED + KIDSDRIV + TRAVTIME + TIF + HOMEKIDS 

row = step_detail[step_detail[0] == '+ HOMEKIDS']
llk_0 = row.iloc[0][2]
df_0 = row.iloc[0][1]
step_summary = step_summary.append(row, ignore_index = True)
X_train = X_train.join(trainData[['HOMEKIDS']])


# In[50]:


step_summary


# ### b)	(5 points).  What predictors does your final model contain?

# The final model contains all predictors.

# ### c)	(5 points).  What are the aliased parameters in your final model?  Please list the predictor’s name and the aliased categories.

# In[ ]:


URBANICITY, MSTATUS, REVOKED, MVR_PTS


# In[ ]:





# ### d)	(5 points).  How many non-aliased parameters are in your final model?

# There is no non-aliased parameters in my final model.

# ### e)	(10 points).  Please show a table of the complete set of parameters of your final model (including the aliased parameters).  Besides the parameter estimates, please also include the standard errors, and the 95% asymptotic confidence intervals.  Conventionally, aliased parameters have missing standard errors and confidence intervals.

# In[52]:


X_interval = trainData[['CAR_AGE', 'MVR_PTS', 'TIF', 'TRAVTIME']]
kidsdriv = trainData[['KIDSDRIV']]
homekids = trainData[['HOMEKIDS']]
X_train = pd.concat([X_interval, term_urban,term_mstatus,term_revoked,term_car_type, kidsdriv, homekids],axis=1)
X_train.insert(0, 'Intercept', 1.0)


outList = Regression.PoissonModel(X_train, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])


# In[55]:


outList[0]


# ## Question 3 (20 points)

# ### You will visually assess your final model in Question 2.  Please color-code the markers according to the Exposure value.

# ### a)	(10 points).  Please plot the predicted number of claims versus the observed number of claims.

# In[57]:



y_pred = outList[6]
plt.figure(dpi = 100)
scatter = plt.scatter(y_train, y_pred, c = o_train)
plt.xlabel('Observed Number of Claims')
plt.ylabel('Predicted Number of Claims')

plt.xticks()
plt.yticks()
plt.grid(axis = 'both')
cbar = plt.colorbar(scatter)
cbar.set_label('EXPOSURE')
plt.show()


# ### b)	(10 points).  Please plot the Deviance residuals versus the observed number of claims.

# In[59]:


r2 =y_train * np.exp(y_pred / y_train) - (y_train - y_pred)
risid = np.where(y_train > y_pred, 1.0, -1.0) * np.where(r2 > 0.0, np.sqrt(2.0 * r2), 0.0)

plt.figure(dpi = 100)
scatter = plt.scatter(y_train, risid, c = o_train)
plt.xlabel('Obserbed Number of Claims')
plt.ylabel('Deviance Residuals')

plt.xticks()
plt.yticks()
plt.grid(axis = 'both')
cbar = plt.colorbar(scatter)
cbar.set_label('EXPOSURE')
plt.show()


# ## Question 4 (20 points)

# ### You will calculate the Accuracy metric to assess your final model in Question 2.
# ### a)	(10 points). Please calculate the Root Mean Squared Error, the Relative Error, and the R-squared metrics.
# 

# In[62]:


# Root mean squared error

y_res = y_train - y_pred
sumRes = np.sum(y_res)
n = len(y_train)
rmse = np.sqrt(np.sum(np.power(y_res, 2)) / n)
rmse


# In[63]:


# Relative Error
mse = np.sum(np.power(y_res, 2)) / n
relError = mse / np.var(y_train)
relError


# In[64]:


# R- squared.
rSqr = np.power(np.corrcoef(y_train, y_pred),2)
rSqr


# ### b)	(10 points). Please comment on the Final Model based on the above three metrics and the diagnostic charts in Question 3.

# Based on the above three matrics, we can see that the RMSE value is low. With relative error value = 0.99, we can conclude that the absolute uncertainty on the model prediction is 0.9947 times the original model. 
# 
# In addition, with the R-Sqared value = 0.059, we can conclude that some part of model is not a good prediction based on the observation. 
# 
# From the diagnostic chart in Question 3 we can see that there are extreme values in the residuals and there are also patterns in the residuals. 
# 
# In conclusion, the final model is not a good fit. 

# In[ ]:




