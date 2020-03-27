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


### Kernel Principal Component Analysis - Kernel PCA

This is used when the data is non-linearly seperable. We map the data to a higher dimension using the Gaussian RBF kernel. We then extract new principle components from there and see how it manages to deal with non-linear problems.

The code below will be used when applying the Kernel PCA.

```
from sklearn.decomposition import KernelPCA
pca = KernelPCA(n_components = 2, kernel = 'rbf')
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
```

**Note**

Unlike the PCA algorithm discussed previously, the below code cannot be used in this case.

```
explained_variance = pca.explained_variance_ratio_
```
We will therefore set n_components = 2 and move on from there.

### Linear Discriminant Analysis - LDA

LDA is a dimensionality reduction technique that is used in the preprocessing step for pattern classification. The goal is to project the dataset under study onto a lower dimensional space. LDA differs from PCA because, in addition to finding the component axises, with LDA we are interested in the axes that maximise the seperation between multiple classes. Therefore, from the n independent variables of the dataset under study, LDA extracts (p less than or equal to n) new independent variables that seperate the most of the classes of the independent variable.

The goal of LDA is to project a feature space (a dataset of n-dimensional samples) onto a smaller subspace k where (k<n), while maintaining the class-discriminatory information.

Both PCA and LDA are linear transformation techniques used for dimensional reduction. PCA is described as unsupervised but LDA is supervised because of the relation to the dependent variable.

**LDA Breakdown**

- Compute the d-dimensional mean vectors for the different classes from the dataset.
- Compute the scatter matrices (in-between-class and within-class scatter matrix).
- Compute the Eigenvectors (e1, e2, ..., eN) and corresponding Eigenvalues (L1, L2, ..., LN), and for the scatter matrices.
- Sort the Eigenvectors by decreasing Eigenvalues and choose k Eigenvectors with the largest Eigenvalues to form a (d * k) dimensional matrix W, where every column represents an Eigenvector.
- Use this (d * k) Eigenvector matrix to transform the samples onto a new subspace. This can be summarised by the matrix multiplication:
       - Y = X * W, where X is a (n * d-dimensional) matrix representing the n samples, and Y are the are the transformed n * k-dimensional samples in the new subspace.
       
The code below will be used when applying the LDA.

```
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
```

Unlike the PCA, we observe that we have fitted X_train and y_train due LDA being a supervised algorithm. In addition, as we are taking 2 linear discriminants we do not require the explained_variance matrix.

### Model Performance Evaluation and Parameter Tuning

After building our machine learning models, some questions remain unanswered:

1) How do we deal with the bias variance tradeoff when building a model and evaluating its performance? The diagram below is used to illustrate this.

<img src = 'Screen1.png' width='700'>


**References**

Bias-Variance tradeoff image: https://towardsdatascience.com/the-bias-variance-tradeoff-8818f41e39e9
