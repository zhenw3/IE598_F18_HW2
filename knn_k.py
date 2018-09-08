# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 05:25:29 2018

@author: zhenw
"""

from sklearn import datasets
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt




#import data
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

print('Class labels:', np.unique(y))

#seperate into training and testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

#standardized the data
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


#trained Knn model k times
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

k_range=range(1,26)
scores=[]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_std, y_train)
    y_pred=knn.predict(X_test_std)
    scores.append(metrics.accuracy_score(y_test,y_pred))

plt.figure

plt.plot(k_range,scores)

plt.xlabel('k',fontsize=15)
plt.ylabel('scores',fontsize=15)
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_24.png', dpi=300)
plt.show()

print("My name is Zhen")
print("My NetID is: zhenw3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")