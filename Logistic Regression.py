#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix


# In[15]:


data = {
    'Feature1': np.random.randn(100),
    'Feature2': np.random.randn(100),
    'Label': np.random.randint(0, 2, 100)
}
dataset = pd.DataFrame(data)


# In[16]:


print("Dataset Sample:")
print(dataset.head())


# In[17]:


X = dataset[['Feature1', 'Feature2']]
y = dataset['Label']


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[19]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[20]:


class LogisticRegressionMDS:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.weights = np.zeros(self.n)
        self.bias = 0

        for _ in range(self.iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            dw = (1 / self.m) * np.dot(X.T, (y_pred - y))
            db = (1 / self.m) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_pred]


# In[21]:


custom_model = LogisticRegressionMDS(learning_rate=0.01, iterations=1000)
custom_model.fit(X_train_scaled, y_train)


# In[22]:


y_pred_custom = custom_model.predict(X_test_scaled)


# In[23]:


accuracy_custom = accuracy_score(y_test, y_pred_custom)
conf_matrix_custom = confusion_matrix(y_test, y_pred_custom)

print(f"\nCustom Logistic Regression Model Accuracy: {accuracy_custom * 100:.2f}%")
print(f"Custom Confusion Matrix:\n{conf_matrix_custom}")


# In[ ]:





# In[ ]:




