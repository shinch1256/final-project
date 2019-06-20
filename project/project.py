#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
import os 

housingprice=load_boston()
print(housingprice.data)


# In[2]:

os.makedirs('plots', exist_ok=True)


housingprice.keys()
print(housingprice.feature_names)
print(housingprice.DESCR)


# In[42]:


#convert to Dataframe for analysis
df=pd.DataFrame(housingprice.data)


# In[43]:


#Change column names from index to actual features
df.columns = housingprice.feature_names
print(df.columns)


# In[44]:


#assign another column to be the price i.e. the target
df['PRICE']=housingprice.target


# In[45]:


df.describe()


# In[46]:


sns.pairplot(df)
plt.savefig('plots/boston_pairplot.png')
plt.clf()


# In[47]:


sns.distplot(df['PRICE'])
plt.savefig('plots/boston_distplot.png')
plt.clf()

# In[56]:


y=df['PRICE']
X=df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT']]
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print('Coefficients: \n',lm.coef_)
print('Intercept: \n',lm.intercept_)


coef = lm.coef_
for i in coef:
    if i>1:
        print(i)
    elif i<-1:
        print(i)
    else:
        print('Not significant')


# In[12]:



predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('predicted_value')
plt.savefig('plots/boston_predict.png')
plt.clf()
#assign a new datafram that includes just the actual values and corresponding predicted value
df2 = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
print(df2)
#pick the first 25 to visualize the difference
df3 = df2.head(25)
df3.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.savefig('plots/boston_scatter.png')
plt.clf()


# In[13]:


#model evaluation

from sklearn import metrics
print('MSE: ',metrics.mean_squared_error(y_test,predictions))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,predictions)))
#residual histogram
sns.distplot((y_test-predictions),bins=50)
plt.title('Residual (difference) Distribution')
plt.savefig('plots/residual_diff.png')
plt.clf()


# In[70]:


#https://github.com/wikipda7/Linear-Regression-Model-for-Boston-Housing-Dataset-Vishal-_files/blob/master/Linear%20Model%20for%20Boston%20Dataset%20(Vishal).ipynb
minimum_price = np.amin(df['PRICE']*21000)
print(minimum_price)
maximum_price = np.amax(df['PRICE']*21000)
print(maximum_price)
mean_price = np.mean(df['PRICE']*21000)
print(mean_price)


# In[76]:


client_data = [[0.03,0,3,0,0.4,5,82,5,2,311,17,396,15]]  # Client 3
print(client_data)
# Show predictions
print(lm.predict(client_data))

    #print "Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price)


# In[ ]:




