#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd            
import numpy as np             
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score,precision_score,classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[2]:


df= pd.read_csv(r"C:\Users\Vishal Kumar\Downloads\WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.head()


#  # Drop unwanted columns

# In[ ]:


df.shape
df.info()
df.isnull().sum().sum()


# In[5]:


df.drop(["EmployeeCount","EmployeeNumber","Over18","StandardHours"],axis=1,inplace=True)
df.head()


# # Encode Target Columnn

# In[10]:


df["Attrition"]=df["Attrition"].map({"Yes":1,"No":0})


# # Encoding Categorial Features

# In[24]:


le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col]=le.fit_transform(df[col])



# # feature vs target

# In[28]:


x = df.drop("Attrition",axis=1)
y = df["Attrition"]


# # Train-Test Split 

# In[66]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state = 42, stratify=y)


# # feature scaling

# In[67]:


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# # Model ( 1- Regression)

# In[68]:


lr = LogisticRegression()
lr.fit(x_train, y_train)

y_pred_lr = lr.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision", precision_score(y_test,y_pred_lr))
print(classification_report(y_test, y_pred_lr))


# # Model(2-Random Forest)

# In[71]:


rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision", precision_score(y_test,y_pred_rf))
print(classification_report(y_test, y_pred_rf))


# # Confusion Matrix

# In[85]:


cm = confusion_matrix(y_test, y_pred_rf)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# # Importance Features

# In[76]:


feature_importance = pd.Series(
    rf.feature_importances_,index=x.columns).sort_values(ascending=False)
feature_importance.head(10)


# In[ ]:




