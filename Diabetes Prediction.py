#!/usr/bin/env python
# coding: utf-8

# **Importing Libraries**

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# **Data Collection and Analysis**

# PIMA Diabetes Dataset

# In[2]:


# Loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('diabetes.csv')


# In[3]:


get_ipython().run_line_magic('pinfo', 'pd.read_csv')


# In[4]:


# Printing the first 5 rows of the dataset
diabetes_dataset.head()


# In[5]:


# number of rows and Columns in this dataset
diabetes_dataset.shape


# In[6]:


# Getting the statistical measures of the data
diabetes_dataset.describe()


# In[7]:


diabetes_dataset['Outcome'].value_counts()


# 0 --> Non-Diabetic
# 
# 1 --> Diabetic

# In[8]:


diabetes_dataset.groupby('Outcome').mean()


# In[9]:


# Seperating the data and labels
X = diabetes_dataset.drop(columns= 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']


# In[10]:


print(X)
print(Y)


# **Data Standardization**

# In[11]:


scaler = StandardScaler()


# In[12]:


scaler.fit(X)


# In[13]:


standardizeddata = scaler.transform(X)


# In[14]:


print(standardizeddata)


# In[15]:


X =standardizeddata
Y = diabetes_dataset['Outcome']


# **Train Test Split**

# In[16]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2, stratify=Y)


# In[17]:


print(X.shape, X_train.shape, X_test.shape)


# **Training the model**

# In[18]:


classfier = svm.SVC(kernel='linear')


# In[19]:


## Training the Support vector Machine Classifier
classfier.fit(X_train, Y_train)


# **Model Evaluation**

# Acccuracy Score

# In[20]:


# Accuracy Score on the training data
X_train_prediction = classfier.predict(X_train)
train_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy for training data: ", train_accuracy*100,"%")


# In[21]:


# Accuracy Score on the test data
X_test_prediction = classfier.predict(X_test)
test_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy for Test data: ", test_accuracy*100,"%")


# **Making a Predictive System**

# In[25]:


input_data = (4,110,92,0,0,37.6,0.191,30)

# Changing the input_data to numpy array
numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
reshaped_data = numpy_array.reshape(1, -1)

# Standardize the input data
std_data = scaler.transform(reshaped_data)

pred = classfier.predict(std_data)
if(pred[0] == 0):
    print("The Person is not diabetic ")
else:
    print("The Person is Diabetic")


# In[ ]:




