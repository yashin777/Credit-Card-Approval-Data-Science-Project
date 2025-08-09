#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install xgboost


# In[3]:


from xgboost import XGBClassifier


# In[4]:


import pandas as pd


# In[5]:


df1=pd.read_csv('application_record.csv')
df2=pd.read_csv('credit_record.csv')


# In[6]:


df1


# In[7]:


df1.duplicated().sum()


# In[8]:


df2.duplicated().sum()


# In[9]:


df2


# In[10]:


combine_df=pd.merge(df1,df2,on='ID', how='outer')


# In[11]:


combine_df


# In[12]:


combine_df=combine_df.drop_duplicates(subset='ID',keep='first')


# In[13]:


combine_df


# In[14]:


combine_df[combine_df['ID']==5150487]


# In[15]:


# There are missing data
combine_df.isnull().mean().sort_values(ascending=False)


# In[33]:


combine_df_columns_to_check=[col for col in combine_df.columns if col not in ['ID', 'STATUS']]
rows_nan=combine_df[combine_df_columns_to_check].isnull().all(axis=1) 
combine_df_clean=combine_df[~rows_nan]


# In[34]:


combine_df_clean


# In[35]:


combine_df=combine_df.dropna()


# In[36]:


from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score


# In[37]:


# I have a multiclass variable I have to Group
def map_status(s):
    if s in ['C','X']:
        return 'Good'
    elif s in ['1','2']:
        return 'Moderate'
    elif s in ['3','4']:
        return 'Bad'
    elif s=='5':
        return 'Very Bad'
    else: 
        return "Unknown"
   



# In[38]:


print(combine_df['STATUS_Grouped'].unique())


# In[39]:


combine_df['STATUS_Grouped']=combine_df['STATUS'].apply(map_status)


# In[40]:


combine_df['STATUS_Grouped']


# In[41]:


combine_df


# In[42]:


le=LabelEncoder()
y=le.fit_transform(combine_df['STATUS_Grouped'])


# In[43]:


X=pd.get_dummies(combine_df.drop(columns=['STATUS','STATUS_Grouped']),drop_first=True)


# In[44]:


X_train,X_test,y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# # Logistic Regression

# In[50]:


logistic_model=LogisticRegression(max_iter=1000)
logistic_model.fit(X_train,y_train)
logistic_prediction=logistic_model.predict(X_test)
print(classification_report(y_test,logistic_prediction))


# # XGBoost

# In[51]:


model=XGBClassifier(
objective='multi:softmax',
    num_class=len(set(y)),
    eval_metrics='mlogloss',
    use_label_encoder=False,
    random_state=42
)
model.fit(X_train,y_train)


# In[52]:


y_predict=model.predict(X_test)


# In[53]:


print("Accuracy:",accuracy_score(y_test,y_pred))


# In[54]:


print("\nClassification Report\n", classification_report(y_test,y_pred))

