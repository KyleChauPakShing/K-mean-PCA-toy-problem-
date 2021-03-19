import numpy as np
from numpy import linalg as LA
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

def create_data(x1, x2, x3):
    x4 = -4.0 * x1
    x5 = 10 * x1 + 10
    x6 = -1 * x2 / 2
    x7 = np.multiply(x2, x2)
    x8 = -1 * x3 / 10
    x9 = 2.0 * x3 + 2.0
    X = np.hstack((x1, x2, x3, x4, x5, x6, x7, x8, x9))
    return X

def pca(X):
    '''
    PCA step by step
      1. normalize matrix X
      2. compute the covariance matrix of the normalized matrix X
      3. do the eigenvalue decomposition on the covariance matrix
    Return: [d, V]
      d is the column vector containing all the corresponding eigenvalues,
      V is the matrix containing all the eigenvectors.
    If you do not remember Eigenvalue Decomposition, please review the linear
    algebra
    In this assignment, we use the ``unbiased estimator'' of covariance. You
    can refer to this website for more information
    http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.cov.html
    Actually, Singular Value Decomposition (SVD) is another way to do the
    PCA, if you are interested, you can google SVD.
    '''

    ####################################################################
    #
    # YOUR CODE HERE!
    #
    ####################################################################

    #Xpca = X
    Xnorm = preprocessing.normalize(X)

    X_input = Xnorm.T
    S = np.cov(X_input)
    #d, V = LA.eig(S)
    eigenValues, eigenVectors = LA.eigh(S)

    idx = eigenValues.argsort()[::-1]   
    d = eigenValues[idx]
    V = eigenVectors[:,idx]

    ####################################################################
    #
    #  Verification using PCA in sklearn
    #
    ####################################################################

    """
    Xpca = preprocessing.normalize(Xpca)
    pcasklearn = PCA()
    pcasklearn.fit(Xpca)

    d_validate = pcasklearn.explained_variance_
    V_validate = pcasklearn.components_

    print('d_validate: ',d_validate)
    print('v_validate',V_validate)
    """
    

    # here d is the column vector containing all the corresponding eigenvalues.
    # V is the matrix containing all the eigenvectors,
    return [d, V]



def plot_figs(X):
    """
    1. perform PCA (you can use pca(X) completed by yourself) on this matrix X
    2. plot (a) The figure of eigenvalues v.s. the order of eigenvalues. All eigenvalues in a decreasing order.
    3. plot (b) The figure of POV v.s. the order of the eigenvalues.
    """


    ####################################################################
    #
    # YOUR CODE HERE!
    #
    ####################################################################

    eigValues, eigVectors = pca(X)

    P1_x_axis_comp = np.arange(1,eigValues.size+1)

    plt.plot(P1_x_axis_comp, eigValues, 'xb-')
    plt.xlabel('The order of Eigenvalue')
    plt.ylabel('Eigenvalues')
    plt.show()

    POV = []
    eigTotal = sum(eigValues)
    tmp = 0.0
    for x in range(eigValues.size):
        tmp = tmp + eigValues[x]
        pov_comp = tmp / eigTotal
        POV.append(pov_comp)

    P2_x_axis_comp = np.arange(1,eigValues.size+1)
    plot2 = plt.figure(2)
    plt.plot(P2_x_axis_comp, POV, 'xr-')
    plt.xlabel('The order of Eigenvalue')
    plt.ylabel('Prop of var.')

    plt.show()

    return

def main():
    N = 1000
    shape = (N, 1)
    x1 = np.random.normal(0, 1, shape) # samples from normal distribution
    #x1 = x1-x1.sum()/1000.0 * np.ones_like(x1)
    x2 = np.random.exponential(10.0, shape) # samples from exponential distribution
    #x2 = x2-x2.sum()/1000.0 * np.ones_like(x2)
    x3 = np.random.uniform(-100, 100, shape) # uniformly sampled data points
    #x3 = x3-x3.sum()/1000.0 * np.ones_like(x3)
    X = create_data(x1, x2, x3)

    print(pca(X))
    plot_figs(X)

if __name__ == '__main__':
    main()

