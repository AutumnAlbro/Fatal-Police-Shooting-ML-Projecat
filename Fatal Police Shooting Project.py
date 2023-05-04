#!/usr/bin/env python
# coding: utf-8

#  # Hypothesis: Is there any correlation between Mental Illness and Police Shootings

# # Import necessary libraries

# In[77]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Read the dataset

# In[78]:


shootings_df = pd.read_csv('fatal-police-shootings-data.csv')


# # Exploratory Data Analysis

# In[79]:



# Remove unnecessary columns
shootings_df.drop(['id', 'name', 'date', 'manner_of_death', 'age', 'gender', 'threat_level', 'flee', 'body_camera'], axis=1, inplace=True)

# Convert categorical data values
shootings_df['race'] = shootings_df['race'].apply(lambda x: 'Non-White' if x != 'W' else 'White')

# Count race values
race_counts = shootings_df['race'].value_counts()
non_white_counts = race_counts['Non-White']

# Create pie chart
labels = ['White', 'Non-White']
sizes = [race_counts['White'], non_white_counts]
colors = ['lightblue', 'orange']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Fatal Police Shootings by Race')
plt.show()




# In[80]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='race', hue='signs_of_mental_illness', data=shootings_df)
plt.title('Race vs. Signs of Mental Illness')
plt.xlabel('Race')
plt.show()


# # Describe the dataframe

# In[81]:


shootings_df.head()


# In[82]:


top_races = shootings_df['race'].value_counts().nlargest(10)
print(top_races)

top_cities = shootings_df['city'].value_counts().nlargest(10)
print(top_cities)


# # Check for missing values

# In[83]:


# check for missing values
print(shootings_df.isnull().sum())


# In[84]:


shootings_df.dropna(inplace=True)


# In[85]:


print(shootings_df.head())


# # Vizualization

# In[86]:


# Create pie chart
labels = ['White', 'Non-White']
sizes = [race_counts['White'], non_white_counts]
colors = ['lightblue', 'orange']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Fatal Police Shootings by Race')
plt.show()


# In[87]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='race', hue='signs_of_mental_illness', data=shootings_df)
plt.title('Race vs. Signs of Mental Illness')
plt.xlabel('Race')
plt.show()


# In[88]:


# Group by race and signs_of_mental_illness and count values
grouped_df = shootings_df.groupby(['race', 'signs_of_mental_illness']).size().reset_index(name='count')

# Create line graph
white_df = grouped_df[grouped_df['race'] == 'White']
non_white_df = grouped_df[grouped_df['race'] == 'Non-White']

plt.plot(white_df['signs_of_mental_illness'], white_df['count'], label='White')
plt.plot(non_white_df['signs_of_mental_illness'], non_white_df['count'], label='Non-White')

plt.xlabel('Signs of Mental Illness')
plt.ylabel('Number of Fatal Shootings')
plt.title('Fatal Police Shootings by Race and Signs of Mental Illness')
plt.legend()
plt.show()


# # Model Support Hypothesis

# In[93]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')


# Correlation analysis
corr_matrix = shootings_df[['race', 'signs_of_mental_illness']].corr()
print(corr_matrix)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(shootings_df[['race', 'signs_of_mental_illness']], shootings_df['armed'], test_size=0.2, random_state=42)

# Encode categorical data
X_train_encoded = pd.get_dummies(X_train)
X_test_encoded = pd.get_dummies(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_encoded, y_train)

# Make predictions on test set
y_pred = model.predict(X_test_encoded)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)


# # Results/Conclusion

# In[94]:


print('When analysing the dataset on police shootings within the US there seems to be some correlation between mental illness and race. The logistic regression model was able to show two variables that could explain if the victim was armed or not. This project only focuses on a subset of variables though meaning it could not full grasp the complexity of police shootings and all the other factors that can be apart of the police shootings. This is why it is important to continue analzying this data to help provent police shootings. With that being said, the conclusion of this project is that the analysis explains the relationship between race and mental illness within police shootings.  ')


# In[ ]:




