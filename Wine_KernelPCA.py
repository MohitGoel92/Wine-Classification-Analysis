# Wine Data Analysis

# Importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset

ds = pd.read_csv('Wine.csv')
X = ds.iloc[:,:-1].values
y = ds.iloc[:,-1].values

# There is no missing data

# Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_X = LabelEncoder()
le_y = LabelEncoder()
y = le_y.fit_transform(y)

# As OneHotEncoder was not used, we do require the "Avoiding the dummy variable" trap step

# Splitting the dataset into the training set and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Dimensionality reduction using Principal Component Analysis (Kernel PCA)

from sklearn.decomposition import KernelPCA
pca = KernelPCA(n_components = 2, kernel = 'rbf')
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Fitting the SVM to the dataset

from sklearn.svm import SVC

# From Grid Search, the best_parameters returned the parameters: C = 10, kernel = linear
# svc = SVC(kernel='rbf')
svc = SVC(kernel = 'linear', C = 10)
svc.fit(X_train, y_train)

# Predicting the test set results

y_pred = svc.predict(X_test)

# Making the confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Evaluating model performance

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(svc, X_train, y_train, cv = 10)
accuracies_avg = accuracies.mean()
accuracies_std = accuracies.std()

# Improving model performance by parameter tuning

from sklearn.model_selection import GridSearchCV

parameters = [{'C':[1,10,100,1000], 'kernel':['linear']},
              {'C':[1,10,100,1000], 'kernel':['rbf'], 'gamma':[0.5, 0.1, 0.01, 0.001]},
              {'C':[1,10,100,1000], 'kernel':['poly'], 'degree':[2,3,4,5], 'gamma':[0.5, 0.1, 0.01, 0.001]}]
grid_search = GridSearchCV(estimator = svc, param_grid = parameters, scoring = 'accuracy', cv=10, n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

# Visualising the training set

from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step = 0.01), 
                     np.arange(start = X_set[:,1].min() - 1, stop = X_set[:,1].max() + 1, step = 0.01))
plt.contourf(X1, X2, svc.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, 
             cmap = ListedColormap(('red','green','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red','green','blue'))(i), label = j)
plt.title('Kernel SVM (Training set)')
plt.legend()
plt.show()

# Visualising the Test set results

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, svc.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, 
             cmap = ListedColormap(('red', 'green','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('Kernel SVM (Test set)')
plt.legend()
plt.show()