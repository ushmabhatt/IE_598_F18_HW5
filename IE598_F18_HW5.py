# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 13:50:09 2018

@author: Ushma
"""

# -*- coding: utf-8 -*-



import pandas as pd
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC
from sklearn import metrics




df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)

# if the Wine dataset is temporarily unavailable from the
# UCI machine learning repository, un-comment the following line
# of code to load the dataset from a local path:

# df_wine = pd.read_csv('wine.data', header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

df_wine.head()

cols = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium','Total phenols',
        'Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue',
        'OD280/OD315 of diluted wines','Proline','Class label']

sns.pairplot(df_wine[cols], size=2.5)
plt.tight_layout()

plt.show()

# Splitting the data into 70% training and 30% test subsets.


cm = np.corrcoef(df_wine[cols].values.T)
#sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=False,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols,
                 xticklabels=cols)

plt.tight_layout()
# plt.savefig('images/10_04.png', dpi=300)
plt.show()

#Box Plot

for i in cols:
    plt.figure()
    sns.boxplot(x=i,data=df_wine)
    #plt.savefig('images/10_02.png', dpi=300)
    
    
#Logistic Regression


X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y,random_state=42)

lr = LogisticRegression()
lr.fit(X_train, y_train)

lr_y_train_pred = lr.predict(X_train)
print( "Logistic Regression train accuracy score: ",metrics.accuracy_score(y_train, lr_y_train_pred) )
lr_y_pred = lr.predict(X_test)
print( "Logistic Regression test accuracy score: ",metrics.accuracy_score(y_test, lr_y_pred) )


svm = SVC(kernel = 'linear', C= 1.0, random_state= 1)
svm.fit(X_train, y_train)
svm_y_train_pred = svm.predict(X_train)
print( "SVM train accuracy score: ",metrics.accuracy_score(y_train, svm_y_train_pred) )
svm_y_pred = svm.predict(X_test)
print("SVM test accuracy score: ",metrics.accuracy_score(y_test, svm_y_pred) )



# Standardizing the data.
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
#Logistic Regression fitted on PCA transformed dataset
lr = LogisticRegression()
lr.fit(X_train_pca, y_train)
pca_lr_y_train_pred = lr.predict(X_train_pca)
pca_lr_y_pred = lr.predict(X_test_pca)
print( "Logistic Regression(PCA) train accuracy score: ",metrics.accuracy_score(y_train, pca_lr_y_train_pred) )
print("Logistic Regression(PCA) test accuracy score: ",metrics.accuracy_score(y_test, pca_lr_y_pred) )




def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],y=X[y == cl, 1],alpha=0.6,c=cmap(idx),edgecolor='black',marker=markers[idx],label=cl)
#
#
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()


#SVM Regression fitted on PCA transformed dataset
pca = PCA(n_components=2)
svm = SVC(kernel = 'linear', C= 1.0, random_state= 1)
svm.fit(X_train_pca, y_train)
pca_svm_y_train_pred = svm.predict(X_train_pca)
pca_svm_y_pred = svm.predict(X_test_pca)
print( "SVM Regression(PCA) train accuracy score: ",metrics.accuracy_score(y_train, pca_svm_y_train_pred) )
print("SVM Regression(PCA) test accuracy score: ",metrics.accuracy_score(y_test, pca_svm_y_pred) )

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],y=X[y == cl, 1],alpha=0.6,c=cmap(idx),edgecolor='black',marker=markers[idx],label=cl)
#
#
plot_decision_regions(X_train_pca, y_train, classifier=svm)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()


#LDA

lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)
lr = LogisticRegression()
lr.fit(X_train_lda, y_train)
#Logistic Regression fitted on LDA transformed dataset
lda_lr_y_train_pred = lr.predict(X_train_lda)
lda_lr_y_pred = lr.predict(X_test_lda)
print( "Logistic Regression(LDA) train accuracy score: ",metrics.accuracy_score(y_train, lda_lr_y_train_pred) )
print("Logistic Regression(LDA) test accuracy score: ",metrics.accuracy_score(y_test, lda_lr_y_pred) )

lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)


lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)

plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_09.png', dpi=300)
plt.show()




X_test_lda = lda.transform(X_test_std)

plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig


#SVM Regression fitted on LDA transformed dataset
svm = SVC(kernel = 'linear', C= 1.0, random_state= 1)
svm.fit(X_train_lda, y_train)
lda_svm_y_train_pred = svm.predict(X_train_lda)
lda_svm_y_pred = svm.predict(X_test_lda)
print( "SVM Regression(LDA) train accuracy score: ",metrics.accuracy_score(y_train, lda_svm_y_train_pred) )
print("SVM Regression(LDA) test accuracy score: ",metrics.accuracy_score(y_test, lda_svm_y_pred) )

plot_decision_regions(X_train_lda, y_train, classifier=svm)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_09.png', dpi=300)
plt.show()


#K-PCA

scikit_kpca = KernelPCA(n_components=2,kernel='rbf', gamma=1)
X_train_skernpca = scikit_kpca.fit_transform(X_train_std, y_train)
X_test_skernpca= scikit_kpca.transform(X_test_std)

#Logistic Regression fitted on KPCA transformed dataset
lr = LogisticRegression()
lr.fit(X_train_skernpca, y_train)
kpca_lr_y_train_pred = lr.predict(X_train_skernpca)
kpca_lr_y_pred = lr.predict(X_test_skernpca)
print( "Logistic Regression(KPCA) train accuracy score(gamma=0.1): ",metrics.accuracy_score(y_train, kpca_lr_y_train_pred) )
print("Logistic Regression(KPCA) test accuracy score(gamma=0.1): ",metrics.accuracy_score(y_test, kpca_lr_y_pred) )
#SVM Regression fitted on KPCA transformed dataset
svm = SVC(kernel = 'linear', C= 1.0, random_state= 1)
svm.fit(X_train_skernpca, y_train)
kpca_svm_y_train_pred = svm.predict(X_train_skernpca)
kpca_svm_y_pred = svm.predict(X_test_skernpca)
print( "SVM Regression(KPCA) train accuracy score(gamma=0.1): ",metrics.accuracy_score(y_train, kpca_svm_y_train_pred) )
print("SVM Regression(KPCA) test accuracy score(gamma=0.1): ",metrics.accuracy_score(y_test, kpca_svm_y_pred) )

lr_train_accu_scores = []
lr_test_accu_scores = []
svm_train_accu_scores = []
svm_test_accu_scores = []


gamma_space = np.linspace(0.01,0.5, endpoint = True)
for gamma in gamma_space:
    scikit_kpca.gamma = gamma
    X_train_skernpca = scikit_kpca.fit_transform(X_train_std, y_train)
    X_test_skernpca= scikit_kpca.transform(X_test_std)
    
    #Logistic Regression fitted on KPCA transformed dataset
    lr = LogisticRegression()
    lr.fit(X_train_skernpca, y_train)
    kpca_lr_y_train_pred = lr.predict(X_train_skernpca)
    kpca_lr_y_pred = lr.predict(X_test_skernpca)
    lr_train_accu_scores.append(metrics.accuracy_score(y_train, kpca_lr_y_train_pred))
    lr_test_accu_scores.append(metrics.accuracy_score(y_test, kpca_lr_y_pred))

    #SVM Regression fitted on KPCA transformed dataset
    svm = SVC(kernel = 'rbf', C= 1.0, random_state= 1)
    svm.fit(X_train_skernpca, y_train)
    kpca_svm_y_train_pred = svm.predict(X_train_skernpca)
    kpca_svm_y_pred = svm.predict(X_test_skernpca)
    svm_train_accu_scores.append(metrics.accuracy_score(y_train, kpca_svm_y_train_pred))
    svm_test_accu_scores.append(metrics.accuracy_score(y_test, kpca_svm_y_pred))


plt.figure()
plt.plot(gamma_space, lr_train_accu_scores, label='Logistic Regression Train Set')
plt.plot(gamma_space, lr_test_accu_scores, label='Logistic Regression Test Set')
plt.legend(loc = 1)
plt.xlabel("Gamma Values")
plt.ylabel("Accuracy Scores")
plt.title("Logistic Regression Fitted on KPCA Transformed Dataset")

plt.figure()
plt.plot(gamma_space, svm_train_accu_scores, label='SVM Regression Train Set')
plt.plot(gamma_space, svm_test_accu_scores, label='SVM Regression Test Set')
plt.legend(loc = 1)
plt.xlabel("Gamma Values")
plt.ylabel("Accuracy Scores")
plt.title("SVM Regression Fitted on KPCA Transformed Dataset")



def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation.
    Parameters
    ------------
    X: {NumPy ndarray}, shape = [n_samples, n_features]
        
    gamma: float
      Tuning parameter of the RBF kernel
        
    n_components: int
      Number of principal components to return
    Returns
    ------------
     X_pc: {NumPy ndarray}, shape = [n_samples, k_features]
       Projected dataset   
    """
    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)

    # Compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    # scipy.linalg.eigh returns them in ascending order
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

    # Collect the top k eigenvectors (projected samples)
    X_pc = np.column_stack((eigvecs[:, i]
                            for i in range(n_components)))

    return X_pc



# ### Example 1: Separating half-moon shapes




X, y = make_moons(n_samples=100, random_state=123)

plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)

plt.tight_layout()
# plt.savefig('images/05_12.png', dpi=300)
plt.show()





scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))

ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1],
              color='red', marker='^', alpha=0.5)
ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1],
              color='blue', marker='o', alpha=0.5)

ax[1].scatter(X_spca[y == 0, 0], np.zeros((50, 1)) + 0.02,
              color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y == 1, 0], np.zeros((50, 1)) - 0.02,
              color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')

plt.tight_layout()
# plt.savefig('images/05_13.png', dpi=300)
plt.show()




X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)

fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))
ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], 
            color='red', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],
            color='blue', marker='o', alpha=0.5)

ax[1].scatter(X_kpca[y==0, 0], np.zeros((50,1))+0.02, 
            color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y==1, 0], np.zeros((50,1))-0.02,
            color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')

plt.tight_layout()
# plt.savefig('images/05_14.png', dpi=300)
plt.show()



# ### Example 2: Separating concentric circles




X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)

plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)

plt.tight_layout()
# plt.savefig('images/05_15.png', dpi=300)
plt.show()




scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))

ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1],
              color='red', marker='^', alpha=0.5)
ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1],
              color='blue', marker='o', alpha=0.5)

ax[1].scatter(X_spca[y == 0, 0], np.zeros((500, 1)) + 0.02,
              color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y == 1, 0], np.zeros((500, 1)) - 0.02,
              color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')

plt.tight_layout()
# plt.savefig('images/05_16.png', dpi=300)
plt.show()




X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
ax[0].scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1],
              color='red', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1],
              color='blue', marker='o', alpha=0.5)

ax[1].scatter(X_kpca[y == 0, 0], np.zeros((500, 1)) + 0.02,
              color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y == 1, 0], np.zeros((500, 1)) - 0.02,
              color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')

plt.tight_layout()
# plt.savefig('images/05_17.png', dpi=300)
plt.show()



# ## Projecting new data points




def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation.
    Parameters
    ------------
    X: {NumPy ndarray}, shape = [n_samples, n_features]
        
    gamma: float
      Tuning parameter of the RBF kernel
        
    n_components: int
      Number of principal components to return
    Returns
    ------------
     alphas: {NumPy ndarray}, shape = [n_samples, k_features]
       Projected dataset 
     
     lambdas: list
       Eigenvalues
    """
    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)

    # Compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    # scipy.linalg.eigh returns them in ascending order
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

    # Collect the top k eigenvectors (projected samples)
    alphas = np.column_stack((eigvecs[:, i]
                              for i in range(n_components)))

    # Collect the corresponding eigenvalues
    lambdas = [eigvals[i] for i in range(n_components)]

    return alphas, lambdas




X, y = make_moons(n_samples=100, random_state=123)
alphas, lambdas = rbf_kernel_pca(X, gamma=15, n_components=1)




x_new = X[25]
x_new




x_proj = alphas[25] # original projection
x_proj




def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new - row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas / lambdas)

# projection of the "new" datapoint
x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)
x_reproj 




plt.scatter(alphas[y == 0, 0], np.zeros((50)),
            color='red', marker='^', alpha=0.5)
plt.scatter(alphas[y == 1, 0], np.zeros((50)),
            color='blue', marker='o', alpha=0.5)
plt.scatter(x_proj, 0, color='black',
            label='original projection of point X[25]', marker='^', s=100)
plt.scatter(x_reproj, 0, color='green',
            label='remapped point X[25]', marker='x', s=500)
plt.legend(scatterpoints=1)

plt.tight_layout()
# plt.savefig('images/05_18.png', dpi=300)
plt.show()



# ## Kernel principal component analysis in scikit-learn




X, y = make_moons(n_samples=100, random_state=123)
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)

plt.scatter(X_skernpca[y == 0, 0], X_skernpca[y == 0, 1],
            color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y == 1, 0], X_skernpca[y == 1, 1],
            color='blue', marker='o', alpha=0.5)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
# plt.savefig('images/05_19.png', dpi=300)
plt.show()

print("My name is Ushma Bhatt")
print("My NetID is: ushmab2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")