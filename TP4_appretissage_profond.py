#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd

data = pd.read_csv('diabetes.csv', sep=",")
print(data)


# In[4]:


import seaborn as sns
sns.pairplot(data)


# In[12]:


from sklearn import tree 
from sklearn.model_selection import train_test_split

X= data[[ 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
          'Insulin' , 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y= data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90, random_state=0) 
 
clf = tree.DecisionTreeClassifier() 
clf.fit(X_train, y_train) 
Z = clf.predict(X_test) 

accuracy = clf.score(X_test,y_test) 
print(accuracy)


# In[3]:


import numpy as np
accuracies = []
Nb_tirage = 100
for i in range(Nb_tirage):
   
  clf = tree.DecisionTreeClassifier() 
  clf.fit(X_train, y_train) 
  Z = clf.predict(X_test) 
  accuracies.append(clf.score(X_test,y_test) )
  #print(accuracies)

print("La moyenne est de:", np.mean(accuracies))
print("La variance est de:", np.var(accuracies))
print("L'écart-type est de:", np.std(accuracies))


# In[4]:


from sklearn.ensemble import BaggingClassifier 
clf = BaggingClassifier(tree.DecisionTreeClassifier(), max_samples=0.5, max_features=0.5, n_estimators=200, random_state=0)

clf.fit(X_train, y_train) 
Z = clf.predict(X_test) 
accuracy=clf.score(X_test,y_test) 
print(accuracy)


# In[5]:


from sklearn.ensemble import BaggingClassifier 
accuraciesb = []
for i in range(100):
   
  clf = BaggingClassifier(tree.DecisionTreeClassifier(), max_samples=0.5,
                          max_features=0.5, n_estimators=200)
  clf.fit(X_train, y_train) 
  Z = clf.predict(X_test) 
  accuraciesb.append(clf.score(X_test,y_test) )
  #print(accuracy)

print("La moyenne est de:", np.mean(accuraciesb))
print("La variance est de:", np.var(accuraciesb))
print("L'écart-type est de:", np.std(accuraciesb))


# In[10]:


import matplotlib.pyplot as plt
n_est= 200
accuracyc = [] #tableau des accuracy

for i in range(1, n_est,15):
  clf = BaggingClassifier(tree.DecisionTreeClassifier(), max_samples=0.5,
                          max_features=0.5, n_estimators=i)
  clf.fit(X_train, y_train) 
  Z = clf.predict(X_test) 
  accuracyc.append(clf.score(X_test,y_test))

plt.plot([i for i in range(1,n_est,15)], accuracyc)
plt.show()


# In[104]:


from sklearn.model_selection import GridSearchCV

tuned_parameters = {'max_samples':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                    'max_features':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
                    }

clf2 = GridSearchCV(BaggingClassifier(tree.DecisionTreeClassifier()),
                    tuned_parameters, cv = 5)

clf2.fit(X_train, y_train)
print(clf2.best_params_)


# In[32]:


from sklearn.ensemble import BaggingClassifier 
accuraciesk = []
for i in range(100):
   
  clf = BaggingClassifier(tree.DecisionTreeClassifier(), max_samples=0.6,
                          max_features=0.7, n_estimators=200)
  clf.fit(X_train, y_train) 
  Z = clf.predict(X_test) 
  accuraciesk.append(clf.score(X_test,y_test) )
  #print(accuracy)

print("La moyenne est de:", np.mean(accuraciesk))
print("La variance est de:", np.var(accuraciesk))
print("L'écart-type est de:", np.std(accuraciesk))


# #random forest

# In[14]:


from sklearn.ensemble import RandomForestClassifier 
clf3 = RandomForestClassifier(n_estimators=200) 
clf3.fit(X_train, y_train) 

y_pred = clf3.predict(X_test) 
accuracy = clf3.score(X_test,y_test) 
print(accuracy)


# In[15]:


accuraciesd = []
Nb_tirage = 100
for i in range(Nb_tirage):
   
  clf4 = RandomForestClassifier(n_estimators=200) 
  clf4.fit(X_train, y_train) 
  Z = clf4.predict(X_test) 
  accuraciesd.append(clf4.score(X_test,y_test) )
  #print(accuracies)

print("La moyenne est de:", np.mean(accuraciesd))
print("La variance est de:", np.var(accuraciesd))
print("L'écart-type est de:", np.std(accuraciesd))


# In[21]:


n_est= 200
accuracye = [] #tableau des accuracy

for i in range(1, n_est, 15):
  clf9 = RandomForestClassifier(n_estimators=i)
  clf9.fit(X_train, y_train) 
  Z = clf9.predict(X_test) 
  accuracye.append(clf9.score(X_test,y_test))

plt.plot([i for i in range(1,n_est, 15)], accuracye)
plt.show()


# In[22]:


from sklearn.ensemble import ExtraTreesClassifier


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90, random_state=0) 
 
    
clf10 = ExtraTreesClassifier(n_estimators = 200)
clf10.fit(X_train, y_train)

acc = clf10.score(X_test, y_test)
print(acc)


# In[26]:


n_est= 200
accuracyg = [] #tableau des accuracy

for i in range(1, n_est,15):
  clf90 = ExtraTreesClassifier(n_estimators=i)
  clf90.fit(X_train, y_train) 
  Z = clf90.predict(X_test) 
  accuracyg.append(clf90.score(X_test,y_test))

plt.plot([i for i in range(1,n_est,15)], accuracyg)
plt.show()


# In[27]:


accuraciesh = []
Nb_tirage = 100
for i in range(Nb_tirage):
   
  clf40 = ExtraTreesClassifier(n_estimators=200) 
  clf40.fit(X_train, y_train) 
  Z = clf40.predict(X_test) 
  accuraciesh.append(clf40.score(X_test,y_test) )
  #print(accuracies)

print("La moyenne est de:", np.mean(accuraciesh))
print("La variance est de:", np.var(accuraciesh))
print("L'écart-type est de:", np.std(accuraciesh))


# In[169]:


from sklearn.ensemble import AdaBoostClassifier 

from sklearn.model_selection import GridSearchCV

for i in range(1,10):
    clf27 = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=i),
    n_estimators=200, learning_rate=2)
    clf27.fit(X_train, y_train)
    accuracy = clf27.score(X_test, y_test)
    print("Le max_depth à", i, "a un accuracy de", accuracy)


# In[173]:


from sklearn.ensemble import AdaBoostClassifier 

from sklearn.model_selection import GridSearchCV

for i in range(1,10):
    clf27 = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=6),
    n_estimators=200, learning_rate=i)
    clf27.fit(X_train, y_train)
    accuracy = clf27.score(X_test, y_test)
    print("Le learning_rate à", i, "a un accuracy de", accuracy)


# In[ ]:




