# Wine Classification Analysis

**Task: Wine Classification**

We are tasked to produce a machine learning model that classifies Wines. We have been given a dataset with 13 variables that are used in classifying the Wines, and it would be ideal to produce a model with fewer variables. If we can produce a model with only 2 variables, we can visually observe the predictive regions and boundaries. In addition, evaluate the model and perform parameter tuning to improve the model.

Therefore, we will be producing a machine learning model to classify Wines using "Dimensionality Reduction", "K-Fold Cross Validation" and "Grid Search" for model evaluation and tuning.

## Dimensionality Reduction

There are two types of Dimensionality Reduction techniques, they are:

- Feature Selection
- Feature Extraction

Feature Selection techniques include "Backward Elimination", "Forward Selection", "Bidirectional Elimination", "Score Comparison" ... etc. For the dataset under study, we will be using Feature Extraction techniques listed below:

- Principal Component Analysis (PCA)
- Kernel Principal Component Analysis (Kernel PCA)
- Linear Discriminant Analysis (LDA)

### Principal Component Analysis - PCA

PCA is one of the most used unsupervised algorithms, and the most popular Dimensionality Reduction Algorithm. PCA is used for operations such as:
- Noise filtering
- Visualisation
- Feature extraction
- Stock market predictions
- Gene data analysis

The goal of PCA is to identify patterns in data and detect the correlation between variables. If we find a strong correlation, we can reduce the dimensionality. In essence, we reduce the dimensions of a d-dimensional dataset by projecting it onto a k-dimensional subspace, where k<d.

**PCA Breakdown**

- Standardise the data.
- Obtain the Eigenvectors and Eigenvalues from the covariance matrix or correlation matrix, or perform Singular Vector Decomposition.
- Sort Eigenvalues in descending order and choose the k Eigenvectors that correspond to the k largest Eigenvalues where k is the number of dimensions of the new feature subspace (k less than or equal to d).
- Construct the projection matrix W from the selected k Eigenvectors.
- Transform the original dataset X via W to obtain a k-dimensional feature subspace Y.

**Summary**

From the m independent variables of our dataset, PCA extracts (p less than or equal to m) new independent variables that explains the most of the variance of the dataset, regardless of the dependent variable. As the dependent variable is not considered, this makes PCA an unsupervised model.

The below code will be used when appling the PCA.

```
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
```

Let's observe the outcome of the explained variance.

```
In [1]: explained_variance
Out[2]: 
array([0.37281068, 0.18739996, 0.10801208, 0.07619859, 0.06261922,
       0.04896412, 0.0417445 , 0.02515945, 0.02340805, 0.0184892 ,
       0.01562956, 0.01269224, 0.00687236])
```

From the output above, we state that 0.56 of the variance (0.37 + 0.19) is contributed by two variables, indicating that we may reduce the dimensionality of the dataset. For the dataset under study, the python files attached take 2 variables as we can visually plot the predictive regions.


