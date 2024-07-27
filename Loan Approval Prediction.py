#!/usr/bin/env python
# coding: utf-8

# # LOAN APPROVAL PRIDICTION

# In[1]:


import pandas as pd


# # USED DATASET

# In[2]:


data = pd.read_csv('loan_prediction.csv')


# # DATASET OVERVIEW

# In[3]:


data


# In[4]:


# Loan_ID : Unique Loan ID

# Gender : Male/ Female

# Married : Applicant married (Y/N)

# Dependents : Number of dependents

# Education : Applicant Education (Graduate/ Under Graduate)

# Self_Employed : Self employed (Y/N)

# ApplicantIncome : Applicant income

# CoapplicantIncome : Coapplicant income

# LoanAmount : Loan amount in thousands of dollars

# Loan_Amount_Term : Term of loan in months

# Credit_History : Credit history meets guidelines yes or no

# Property_Area : Urban/ Semi Urban/ Rural

# Loan_Status : Loan approved (Y/N) this is the target variable


# In[5]:


data.head()


# In[6]:


data.tail()


# In[7]:


data.shape


# In[8]:


print("Number of Rows",data.shape[0])
print("Number of Columns",data.shape[1])


# # DATASET INFORMATION 

# In[9]:


data.info()


# In[10]:


data.isnull().sum()


# # CHEAK NULL VALUE IN DATASET

# In[11]:


data.isnull().sum()*100 / len(data)


# In[12]:


data = data.drop('Loan_ID',axis=1)


# In[13]:


data.head(1)


# In[14]:


columns = ['Gender','Dependents','LoanAmount','Loan_Amount_Term']


# In[15]:


data = data.dropna(subset=columns)


# In[16]:


data.isnull().sum()*100 / len(data)


# In[17]:


data['Self_Employed'].mode()[0]


# In[18]:


data['Self_Employed'] =data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])


# In[19]:


data.isnull().sum()*100 / len(data)


# In[20]:


data['Gender'].unique()


# In[21]:


data['Self_Employed'].unique()


# In[22]:


data['Credit_History'].mode()[0]


# In[23]:


data['Credit_History'] =data['Credit_History'].fillna(data['Credit_History'].mode()[0])


# In[24]:


data.isnull().sum()*100 / len(data)


# In[25]:


data.sample(5)


# In[26]:


data['Dependents'] =data['Dependents'].replace(to_replace="3+",value='4')


# In[27]:


data['Dependents'].unique()


# In[28]:


data['Loan_Status'].unique()


# In[29]:


data['Gender'] = data['Gender'].map({'Male':1,'Female':0}).astype('int')
data['Married'] = data['Married'].map({'Yes':1,'No':0}).astype('int')
data['Education'] = data['Education'].map({'Graduate':1,'Not Graduate':0}).astype('int')
data['Self_Employed'] = data['Self_Employed'].map({'Yes':1,'No':0}).astype('int')
data['Property_Area'] = data['Property_Area'].map({'Rural':0,'Semiurban':2,'Urban':1}).astype('int')
data['Loan_Status'] = data['Loan_Status'].map({'Y':1,'N':0}).astype('int')


# # DATA PREPROCESSING 

# In[30]:


data.head()


# In[31]:


X = data.drop('Loan_Status',axis=1)


# In[32]:


y = data['Loan_Status']


# In[33]:


y


# In[34]:


data.head()


# In[35]:


cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']


# In[36]:


from sklearn.preprocessing import StandardScaler
st = StandardScaler()
X[cols]=st.fit_transform(X[cols])


# In[37]:


X


# # Splitting The Dataset Into The Training Set And Test Set & Applying K-Fold Cross Validation

# In[38]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np


# In[39]:


model_df={}
def model_val(model,X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,
                                                   test_size=0.20,
                                                   random_state=42)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print(f"{model} accuracy is {accuracy_score(y_test,y_pred)}")
    
    score = cross_val_score(model,X,y,cv=5)
    print(f"{model} Avg cross val score is {np.mean(score)}")
    model_df[model]=round(np.mean(score)*100,2)


# In[40]:


model_df


# # 1.Logistic Regression 

# In[41]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model_val(model,X,y)


# # 2.SVC

# In[42]:


from sklearn import svm
model = svm.SVC()
model_val(model,X,y)


# # 3.Decision Tree Classifier

# In[43]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model_val(model,X,y)


# # 4.Random Forest Classifier

# In[44]:


from sklearn.ensemble import RandomForestClassifier
model =RandomForestClassifier()
model_val(model,X,y)


# # 5.Gradient Boosting Classifier

# In[45]:


from sklearn.ensemble import GradientBoostingClassifier
model =GradientBoostingClassifier()
model_val(model,X,y)


# # Hyperparameter Tuning

# In[46]:


from sklearn.model_selection import RandomizedSearchCV


# # 1.Logistic Regression

# In[47]:


log_reg_grid={"C":np.logspace(-4,4,20),
             "solver":['liblinear']}


# In[48]:


rs_log_reg=RandomizedSearchCV(LogisticRegression(),
                   param_distributions=log_reg_grid,
                  n_iter=20,cv=5,verbose=True)


# In[49]:


rs_log_reg.fit(X,y)


# In[50]:


rs_log_reg.best_score_


# In[51]:


rs_log_reg.best_params_


# # 2.SVC

# In[52]:


svc_grid = {'C':[0.25,0.50,0.75,1],"kernel":["linear"]}


# In[53]:


rs_svc=RandomizedSearchCV(svm.SVC(),
                  param_distributions=svc_grid,
                   cv=5,
                   n_iter=20,
                  verbose=True)


# In[54]:


rs_svc.fit(X,y)


# In[55]:


rs_svc.best_score_


# In[56]:


rs_svc.best_params_


# # 3.Random Forest Classifier

# In[57]:


RandomForestClassifier()


# In[58]:


rf_grid={'n_estimators':np.arange(10,1000,10),
  'max_features':['auto','sqrt'],
 'max_depth':[None,3,5,10,20,30],
 'min_samples_split':[2,5,20,50,100],
 'min_samples_leaf':[1,2,5,10]
 }


# In[59]:


rs_rf=RandomizedSearchCV(RandomForestClassifier(),
                  param_distributions=rf_grid,
                   cv=5,
                   n_iter=20,
                  verbose=True)


# In[60]:


rs_rf.fit(X,y)


# In[61]:


rs_rf.best_score_


# In[62]:


rs_rf.best_params_


# # Save The Model

# In[63]:


X = data.drop('Loan_Status',axis=1)
y = data['Loan_Status']


# In[64]:


rf = RandomForestClassifier(n_estimators=270,
 min_samples_split=5,
 min_samples_leaf=5,
 max_features='sqrt',
 max_depth=5)


# In[65]:


rf.fit(X,y)


# In[66]:


import joblib


# In[67]:


joblib.dump(rf,'loan_status_predict')


# In[68]:


model = joblib.load('loan_status_predict')


# In[69]:


import pandas as pd
df = pd.DataFrame({
    'Gender':1,
    'Married':1,
    'Dependents':2,
    'Education':0,
    'Self_Employed':0,
    'ApplicantIncome':2889,
    'CoapplicantIncome':0.0,
    'LoanAmount':45,
    'Loan_Amount_Term':180,
    'Credit_History':0,
    'Property_Area':1
},index=[0])


# In[70]:


df


# In[71]:


result = model.predict(df)


# In[72]:


if result==1:
    print("Loan Approved")
else:
    print("Loan Not Approved")


# # GUI

# In[73]:


from tkinter import *
import joblib
import pandas as pd


# In[ ]:


import tkinter as tk
from tkinter import Label, Entry, Button
import joblib
import pandas as pd

def show_entry():
    p1 = float(e1.get())
    p2 = float(e2.get())
    p3 = float(e3.get())
    p4 = float(e4.get())
    p5 = float(e5.get())
    p6 = float(e6.get())
    p7 = float(e7.get())
    p8 = float(e8.get())
    p9 = float(e9.get())
    p10 = float(e10.get())
    p11 = float(e11.get())

    model = joblib.load('loan_status_predict')
    df = pd.DataFrame({
        'Gender': p1,
        'Married': p2,
        'Dependents': p3,
        'Education': p4,
        'Self_Employed': p5,
        'ApplicantIncome': p6,
        'CoapplicantIncome': p7,
        'LoanAmount': p8,
        'Loan_Amount_Term': p9,
        'Credit_History': p10,
        'Property_Area': p11
    }, index=[0])
    result = model.predict(df)

    if result == 1:
        Label(master, text="Loan approved", font=('Helvetica', 14)).grid(row=31)
    else:
        Label(master, text="Loan Not Approved", font=('Helvetica', 14)).grid(row=31)

def refresh():
    e1.delete(0, tk.END)
    e2.delete(0, tk.END)
    e3.delete(0, tk.END)
    e4.delete(0, tk.END)
    e5.delete(0, tk.END)
    e6.delete(0, tk.END)
    e7.delete(0, tk.END)
    e8.delete(0, tk.END)
    e9.delete(0, tk.END)
    e10.delete(0, tk.END)
    e11.delete(0, tk.END)

master = tk.Tk()
master.title("Loan Status Prediction Using Machine Learning")
label = Label(master, text="Loan Status Prediction", bg="black", fg="white", font=('Helvetica', 16)).grid(row=0, columnspan=2)

Label(master, text="Gender [1:Male, 0:Female]", font=('Helvetica', 12)).grid(row=1)
Label(master, text="Married [1:Yes, 0:No]", font=('Helvetica', 12)).grid(row=2)
Label(master, text="Dependents [1, 2, 3, 4]", font=('Helvetica', 12)).grid(row=3)
Label(master, text="Education", font=('Helvetica', 12)).grid(row=4)
Label(master, text="Self_Employed", font=('Helvetica', 12)).grid(row=5)
Label(master, text="ApplicantIncome", font=('Helvetica', 12)).grid(row=6)
Label(master, text="CoapplicantIncome", font=('Helvetica', 12)).grid(row=7)
Label(master, text="LoanAmount", font=('Helvetica', 12)).grid(row=8)
Label(master, text="Loan_Amount_Term", font=('Helvetica', 12)).grid(row=9)
Label(master, text="Credit_History", font=('Helvetica', 12)).grid(row=10)
Label(master, text="Property_Area", font=('Helvetica', 12)).grid(row=11)

e1 = Entry(master, font=('Helvetica', 12))
e2 = Entry(master, font=('Helvetica', 12))
e3 = Entry(master, font=('Helvetica', 12))
e4 = Entry(master, font=('Helvetica', 12))
e5 = Entry(master, font=('Helvetica', 12))
e6 = Entry(master, font=('Helvetica', 12))
e7 = Entry(master, font=('Helvetica', 12))
e8 = Entry(master, font=('Helvetica', 12))
e9 = Entry(master, font=('Helvetica', 12))
e10 = Entry(master, font=('Helvetica', 12))
e11 = Entry(master, font=('Helvetica', 12))

e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)
e8.grid(row=8, column=1)
e9.grid(row=9, column=1)
e10.grid(row=10, column=1)
e11.grid(row=11, column=1)

Button(master, text="Predict", command=show_entry, font=('Helvetica', 12)).grid(row=12, column=0, pady=10)
Button(master, text="Refresh", command=refresh, font=('Helvetica', 12)).grid(row=12, column=1, pady=10)

master.mainloop()


# In[ ]:





# In[ ]:




