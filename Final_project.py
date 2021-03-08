#!/usr/bin/env python
# coding: utf-8

# ### Importing all the required libraries

# In[92]:


import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, StratifiedKFold
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC


import tensorflow as tf
import tensorflow.keras as k

import plotly.graph_objects as go
from plotly.graph_objects import FigureWidget
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')


# In[2]:


df = pd.read_csv('train(1).csv')
df.drop('id', axis = 1, inplace = True)
df.shape


# In[3]:


# Counting the number of Categorical and Binary features
cats = 0
bins = 0

for i in df.columns:
    if 'cat' in i.split('_'):
        cats += 1
    elif 'bin' in i.split('_'):
        bins += 1
print('categorical features:', cats)
print('binary features:', bins)


# In[4]:


# Finding if the target column is balanced or not
df.target.value_counts()


# In[5]:


# Counting the missing values
def count_nulls(df):
    '''Function to count the total number of missing values in the dataset'''
    missing = {}
    for i in df.columns:
        count = 0
        for j in df[i]:
            if j == -1:
                count += 1
        missing[i] = count
    return pd.Series(missing)

missing = count_nulls(df)

missing_col_list = [i for i in missing.keys() if missing[i] != 0]

print('No. of features with missing values:', len(missing_col_list))


# In[6]:


# Top 2 features with high number of missing values
missing.nlargest(2)/len(df)


# In[7]:


missing.nlargest(12)


# In[8]:


''''
- Different ways to deal with the missing values
- Starting with the bottom of the list
- For numerical values, they can be replaced by median of the data, or the row can be removed if the no. of missing values is too low
    - For the last 6 missing value features, the values will be imputed using a simple imputer
    - Median for numerical values and most frequent for categorical values
'''


# In[9]:


num_impute = SimpleImputer(missing_values = -1, strategy = 'median')
cat_impute = SimpleImputer(missing_values = -1, strategy = 'most_frequent')


# In[10]:


# Creating a copy dataframe for doing any changes into
data = df.copy()


# In[11]:


data.ps_car_12 = num_impute.fit_transform(data[['ps_car_12']])
data.ps_car_11 = num_impute.fit_transform(data[['ps_car_11']])


# In[12]:


for i in missing.nlargest(12)[-6:-2].keys():
    data[i] = cat_impute.fit_transform(data[[i]])


# In[13]:


'''
The features with the top 3 highest number of missing values will be removed from the feature list as the number of missing values is too high
'''


# In[14]:


data.drop(missing.nlargest(12)[0:3].keys(), axis = 1, inplace = True)


# In[15]:


count_nulls(data).nlargest(5)


# In[16]:


data.ps_car_07_cat.value_counts()/len(df), data.ps_ind_05_cat.value_counts()/len(df)
# Removing these two features as over 85% of data is one category and hence they don't have enough variation to affect the final outcome


# In[17]:


data.drop(['ps_car_07_cat', 'ps_ind_05_cat'], axis = 1, inplace = True)


# In[18]:


count_nulls(data).nlargest(2)


# In[19]:


# Replacing the null value in the ps_car_14 column with the median
data.ps_car_14 = num_impute.fit_transform(data[['ps_car_14']])


# In[20]:


'''All missing values have been fixed and the data has no more missing values'''


# In[ ]:





# In[21]:


'''Visulisation of the distribution of values in each feature'''


# In[22]:


data.columns


# In[23]:


cat_feats = [i for i in data.columns if 'cat' in i.split('_')]
bin_feats = [i for i in data.columns if 'bin' in i.split('_')]

print(len(cat_feats), len(bin_feats))


# In[24]:


# Visualising the distribution of values in bin_feats


# In[25]:


feats_to_remove = []

for i in bin_feats:
    # Adding the features in a list that will be removed because they have over 90% of the same value
    counts = data[i].value_counts()/len(data)
    worst = counts.nlargest(1)
    if worst.values >= 0.9:
        feats_to_remove.append(i)
    
    print(i)
    fig = go.FigureWidget(data = go.Bar(x = counts.keys(), y = counts.values))
    fig.show()

for i in cat_feats:
    counts = data[i].value_counts()/len(data)
    worst = counts.nlargest(1)
    if worst.values >= 0.9:
        feats_to_remove.append(i)
    print(i)
    fig = go.FigureWidget(data = go.Bar(x = counts.keys(), y = counts.values))
    fig.show()


# In[26]:


feats_to_remove


# In[27]:


# From the above two set of plots for binary and categorical features, we can select the features that have very low variance
data.drop(feats_to_remove, axis = 1, inplace = True)


# In[28]:


'''Perfoming similar type of analysis with the numeric features in the data'''


# In[29]:


data.columns


# In[30]:


numeric_feats = [i for i in data.columns if ('cat' not in i.split('_')) and ('bin' not in i.split('_'))]
'''These features, apart from the target, represent either ordinal values or interval values
We will try to see their distribution against the target variable'''
numeric_feats, len(numeric_feats)


# In[31]:


for i in numeric_feats[1:]:
    sns.distplot(data[i])
    plt.title(i)
    plt.show()
#     fig = go.FigureWidget(data = go.Histogram(x = data[i]))
#     fig.show()


# In[32]:


# The feature ps_ind_14 has almost no variation in the data, so it can be removed from the feature list
data.ps_ind_14.value_counts()/len(data)


# In[33]:


data.drop('ps_ind_14', axis = 1, inplace = True)


# In[ ]:





# In[34]:


cat_feats = [i for i in data.columns if 'cat' in i.split('_')]
bin_feats = [i for i in data.columns if 'bin' in i.split('_')]
calc_feats = [i for i in data.columns if 'calc' in i.split('_')]
remaining = [i for i in data.columns if i not in cat_feats + bin_feats + calc_feats]


# In[35]:


len(cat_feats), len(bin_feats), len(calc_feats), len(remaining)


# In[36]:


# Categorical Features
plt.figure(figsize = (15, 15))
for i in range(len(cat_feats)):
    plt.subplot(3, 3, i+1)
    plt.ylabel('Count')
    sns.countplot(data[cat_feats[i]])


# In[37]:


# Binary Features
plt.figure(figsize = (20, 20))
for i in range(len(bin_feats)):
    plt.subplot(5, 3, i+1)
    plt.ylabel('Count')
    sns.countplot(data[bin_feats[i]])


# In[38]:


# Calculated Features
plt.figure(figsize = (20, 20))
for i in range(len(calc_feats)):
    plt.subplot(5, 4, i+1)
    
    if i < 3:
        plt.ylabel('Count')
        sns.distplot(data[calc_feats[i]])
    else:    
        plt.ylabel('Count')
        sns.countplot(data[calc_feats[i]])


# In[39]:


data[remaining]

# Remaining Features
plt.figure(figsize = (20, 20))
for i in range(1, len(remaining)):
    plt.subplot(3, 4, i+1)
    
    if i > 6:
        plt.ylabel('Count')
        sns.distplot(data[remaining[i]])
    else:    
        plt.ylabel('Count')
        sns.countplot(data[remaining[i]])


# In[40]:


plt.figure(figsize = (8, 8))
sns.heatmap(np.around(data[cat_feats].corr(), 2), annot = True, square = True)


# In[41]:


plt.figure(figsize = (8, 8))
sns.heatmap(np.around(data[bin_feats].corr(), 2), annot = True, square = True)


# In[42]:


plt.figure(figsize = (8, 8))
sns.heatmap(np.around(data[calc_feats].corr(), 2), annot = True, square = True)


# In[43]:


plt.figure(figsize = (8, 8))
sns.heatmap(np.around(data[remaining].corr(), 2), annot = True, square = True)


# ### Summary of EDA & EDA Deliverables:
# - Data is not balanced, there is a huge mismatch between the two output values 0 & 1
# - The data can be balanced by taking all the values from the lower category and then sampling the other category to match the no. of samples of the lower category
# - Keeping all the values where target = 1 and sampling 413959 values of target = 0 will lead to a 12% balance in the data
# - There are 14 Categorical and 17 Binary features
# - Top 2 features in terms of missing values: 'ps_car_03_cat', 'ps_car_05_cat'
# - Total 12 features have missing values
# - Features suitable for Standard Scaler: 'ps_car_12', 'ps_car_13', 'ps_car_14'
# - Almost all the Binary features are dominated by one value
# - Several categorical features are also dominated by a single category
# - Several features were dropped because pretty much the entire feature had a single value like: ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_car_10_cat', 'ps_car_07_cat', 'ps_ind_05_cat', 'ps_ind_14'
# - Features that had a huge number of missing values were removed from the data: 'ps_car_03_cat', 'ps_car_05_cat'
# - Features that had small number of missing values were filled according to either the Median or the Most frequent class
# - None of the Categorical, Binary, Ordinal or Interval features show strong correlation with each other or with the target variable

# ### Modelling

# In[44]:


onehot = OneHotEncoder()
one = onehot.fit_transform(data[cat_feats]).toarray()


# In[45]:


std_scale = StandardScaler()
data[['ps_car_12', 'ps_car_13', 'ps_car_14']] = std_scale.fit_transform(data[['ps_car_12', 'ps_car_13', 'ps_car_14']])


# In[46]:


y = data[['target']]


# In[47]:


new_data = data[[i for i in data.columns if (i not in cat_feats) and (i != 'target')]]
new_data = np.array(new_data)
new_data = np.hstack((new_data, one))


# In[48]:


# One Hot Encoded train test split data
x_train, x_test, y_train, y_test = train_test_split(new_data, y, test_size = 0.2, stratify = y)


# In[55]:


# Non-encoded train test split data
x_train_ne, x_test_ne, y_train_ne, y_test_ne = train_test_split(data.drop('target', axis = 1), y, test_size = 0.2, stratify = y)


# In[77]:


# Creating a balanced datset
balanced_data = data[data.target == 1]
balanced_data = pd.concat([balanced_data, data[data.target == 0].sample(n = len(balanced_data))])


# In[78]:


# Non-encoded train test split data on Balanced dataset
x_train_b, x_test_b, y_train_b, y_test_b = train_test_split(balanced_data.drop('target', axis = 1), balanced_data.target, test_size = 0.2, stratify = balanced_data.target)


# In[57]:


x_train_ne.shape[1] - x_train.shape[1]


# There is an increase of 150 features in the One Hot Encoded data

# ### Logistic Regression

# #### Using One Hot Encoded data

# In[49]:


logreg = LogisticRegression(n_jobs = -1)
logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)


# In[50]:


print(metrics.accuracy_score(y_test, y_pred))
print(metrics.precision_score(y_test, y_pred))
print(metrics.recall_score(y_test, y_pred))


# In[51]:


np.unique(y_pred)


# The Logistic Regression gives high accuracy because of the imbalanced dataset. This is a prime example where high accuracy doesn't mean a good model. In fact, it is a very bad model because the model only predicts 0 as the result. Since, the highly imbalanced dataset has over 95% 0 values, the accuracy is high but the recall, precision and F-score are all 0.

# #### Using non-encoded data

# In[58]:


logreg = LogisticRegression(n_jobs = -1)
logreg.fit(x_train_ne, y_train_ne)

y_pred = logreg.predict(x_test_ne)


# In[59]:


print(metrics.accuracy_score(y_test_ne, y_pred))
print(metrics.precision_score(y_test_ne, y_pred))
print(metrics.recall_score(y_test_ne, y_pred))


# No improvement is observed after training on a Non-encoded dataset. Only the training time has been reduced.

# ### Linear SVC

# #### Using One Hot Encoded data

# In[52]:


svc = LinearSVC(max_iter = 10000, dual = False)
svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)


# The LinearSVC was not able to fit even with 10000 iterations with dual enabled. However, the documentation says to make it False in case of n_samples > n_features

# In[53]:


print(metrics.accuracy_score(y_test, y_pred))
print(metrics.precision_score(y_test, y_pred))
print(metrics.recall_score(y_test, y_pred))


# In[54]:


np.unique(y_pred)


# #### Using non-encoded data

# In[81]:


svc = LinearSVC(max_iter = 10000, dual = False)
svc.fit(x_train_ne, y_train_ne)

y_pred = svc.predict(x_test_ne)


# In[82]:


print(metrics.accuracy_score(y_test_ne, y_pred))
print(metrics.precision_score(y_test_ne, y_pred))
print(metrics.recall_score(y_test_ne, y_pred))


# No improvement is observed after training on a Non-encoded dataset. Only the training time has been reduced.

# #### Non-encoded balanced dataset

# In[83]:


svc = LinearSVC(max_iter = 10000, dual = False)
svc.fit(x_train_b, y_train_b)

y_pred = svc.predict(x_test_b)


# In[85]:


print(metrics.accuracy_score(y_test_b, y_pred))
print(metrics.precision_score(y_test_b, y_pred))
print(metrics.recall_score(y_test_b, y_pred))
print(metrics.f1_score(y_test_b, y_pred))


# We definitely see an improvement in the model's predictions on a balanced dataset.

# ### AdaBoost Classifier

# In[176]:


ada = AdaBoostClassifier()
ada.fit(x_train_ne, y_train_ne)


# In[177]:


y_pred = ada.predict(x_test_ne)


# In[178]:


print(metrics.accuracy_score(y_test_ne, y_pred))
print(metrics.precision_score(y_test_ne, y_pred))
print(metrics.recall_score(y_test_ne, y_pred))
print(metrics.f1_score(y_test_ne, y_pred))


# In[179]:


np.unique(y_pred, return_counts = True), np.unique(y_test, return_counts = True)


# ### XGBoost

# In[172]:


xgb_classifier = xgb.XGBClassifier(missing = None)
xgb_classifier.fit(x_train_ne, y_train_ne)


# In[173]:


y_pred = xgb_classifier.predict(x_test_ne)


# In[174]:


print(metrics.accuracy_score(y_test_ne, y_pred))
print(metrics.precision_score(y_test_ne, y_pred))
print(metrics.recall_score(y_test_ne, y_pred))
print(metrics.f1_score(y_test_ne, y_pred))


# In[175]:


np.unique(y_pred, return_counts = True), np.unique(y_test, return_counts = True)


# - The AdaBoost Classifier does not give better results than than the XGBoost Classifier
# - Better score can be achieved with XGBoost if more time is spent tuning the parameters

# ### MLPClassifier or Keras Based Classifier

# We will be using the NN built using Keras to perform this task rather than MLPClassifier provided by the Scikit-learn for GPU usage and better control over our network.

# In[154]:


def create_model():
    ins = k.Input(shape = (x_train.shape[1]))
    
    x = k.layers.Dense(1024, activation = 'relu')(ins)
    x = k.layers.BatchNormalization()(x)
    
    for i in range(15):
        x = k.layers.Dense(512, activation = 'relu')(x)
        x = k.layers.BatchNormalization()(x)
    
    x = k.layers.Dense(256, activation = 'relu')(x)
    x = k.layers.BatchNormalization()(x)
    
    x = k.layers.Dense(128, activation = 'relu')(x)
    x = k.layers.BatchNormalization()(x)
    
    x = k.layers.Dense(64, activation = 'relu')(x)
    x = k.layers.BatchNormalization()(x)
    
    outs = k.layers.Dense(1, activation = 'sigmoid')(x)
    
    model = k.Model(inputs = ins, outputs = outs)
    metrics = [k.metrics.Precision(thresholds = 0.5), k.metrics.Recall(thresholds = 0.5), 'acc']
    opt = k.optimizers.Adam(learning_rate = 0.001)
    model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = metrics)
    return model


# In[155]:


model = create_model()
model.summary()


# In[158]:


calls = k.callbacks.ReduceLROnPlateau(patience = 3)
history = model.fit(x = x_train, y = y_train, batch_size = 8192, validation_data = (x_test, y_test), epochs = 50, callbacks = calls)


# In[159]:


y_pred = model.predict(x_test)


# In[160]:


for i, val in enumerate(y_pred):
    if val <= 0.5:
        y_pred[i] = 0
    else:
        y_pred[i] = 1


# In[161]:


np.unique(y_pred, return_counts = True)


# In[162]:


print(metrics.accuracy_score(y_test, y_pred))
print(metrics.precision_score(y_test, y_pred))
print(metrics.recall_score(y_test, y_pred))
print(metrics.f1_score(y_test, y_pred))


# In[164]:


len(model.layers)


# Final Observations:
# - The best F1-score is achieved at a layer size of 40, including the Batch Normalisation, Input and Output layers for the Keras model
# - The best model for this case is Keras model
# - For both the AdaBoost Classifier and the XGBoost Classifier, there is no improvement by encoding the features which is to be expected as they are both using Decision Trees as their base estimator
# - Keras model is better if not missing a positive sample  is the priority as the recall is highest
# - XGBoost model is better if not marking negative samples as positive is priority as the precision is the highest
