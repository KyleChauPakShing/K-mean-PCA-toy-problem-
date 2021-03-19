import os.path
import numpy as np
import matplotlib.pyplot as plt
import imageio # updated
from sklearn.decomposition import PCA
from sklearn import preprocessing
from numpy import linalg as LA

def load_data(digits, num):
    '''
    Loads all of the images into a data-array.

    The training data has 5000 images per digit,
    but loading that many images from the disk may take a while.  So, you can
    just use a subset of them, say 300 for training (otherwise it will take a
    long time to complete.

    Note that each image as a 28x28 grayscale image, loaded as an array and
    then reshaped into a single row-vector.

    Use the function display(row-vector) to visualize an image.

    '''
    totalsize = 0
    for digit in digits:
        totalsize += min([len(next(os.walk('train%d' % digit))[2]), num])
    print('We will load %d images' % totalsize)
    X = np.zeros((totalsize, 784), dtype = np.uint8)   #784=28*28
    for index in range(0, len(digits)):
        digit = digits[index]
        print('\nReading images of digit %d' % digit)
        for i in range(num):
            pth = os.path.join('train%d' % digit,'%05d.pgm' % i)
            image = imageio.imread(pth).reshape((1, 784))  # updated
            X[i + index * num, :] = image
        print('\n')
    return X

def plot_mean_image(X, digits = [0]):
    ''' example on presenting vector as an image
    '''
    plt.close('all')
    meanrow = X.mean(0)
    # present the row vector as an image
    plt.imshow(np.reshape(meanrow,(28,28)))
    plt.title('Mean image of digit ' + str(digits))
    plt.gray(), plt.xticks(()), plt.yticks(()), plt.show()

def pca(X):

    # From ex1, we learned how to perform PCA on a given input
    Xnorm = preprocessing.normalize(X)
    X_input = Xnorm.T
    S = np.cov(X_input)
    #d, V = LA.eig(S)
    eigenValues, eigenVectors = LA.eigh(S)

    idx = eigenValues.argsort()[::-1]
    d = eigenValues[idx]
    V = eigenVectors[:,idx]
    ### The first column has the eigenvector correspond to the largest eigenValue
    ### more info on https://towardsdatascience.com/pca-with-numpy-58917c1d0391 

    ### This time we only return the 9 eigenvector that we need:
    Tmp = []
    for x in range(9):
        Tmp.append(V[:,x])
    TmpArr = np.array(Tmp)
    TmpArr = np.reshape(TmpArr,(9,28,28))
    V = TmpArr

    return [d, V]

def plot_eigen_image(EigenImagesArray):

    plt.close('all')
    plt.figure(num='EigenImages')  #创建一个名为astronaut的窗口,并设置大小 
    for x in range(0,9):
        plt.subplot(3,3,(x+1))
        plt.title('Eigen image '+ str(x+1))
        plt.gray(), plt.xticks(()), plt.yticks(())
        plt.imshow(EigenImagesArray[x])

    plt.savefig('eigenimages.jpg')

def plot_pov(EigenValuesArray):

    POV = []
    eigTotal = sum(EigenValuesArray)
    tmp = 0.0
    for x in range(EigenValuesArray.size):
        tmp = tmp + EigenValuesArray[x]
        pov_comp = tmp / eigTotal
        POV.append(pov_comp)

    P2_x_axis_comp = np.arange(1,EigenValuesArray.size+1)
    plot2 = plt.figure(2)
    plt.plot(P2_x_axis_comp, POV, 'xr-')
    plt.xlabel('The order of Eigenvalue')
    plt.ylabel('Prop of var.')
    plt.savefig('pov.jpg')


def main():
    digits = [0, 1, 2]
    # load handwritten images of digit 0, 1, 2 into a matrix X
    # for each digit, we just use 300 images
    # each row of matrix X represents an image
    X = load_data(digits, 300)


    # plot the mean image of these images!
    # you will learn how to represent a row vector as an image in this function
    plot_mean_image(X, digits)

    ####################################################################
    # you need to
    #   1. do the PCA on matrix X;
    #
    #   2. plot the eigenimages (reshape the vector to 28*28 matrix then use
    #   the function ``imshow'' in pyplot), save the images of eigenvectors
    #   which correspond to largest 9 eigenvalues. Save them in a single file
    #   ``eigenimages.jpg''.
    #
    #   3. plot the POV (the Portion of variance explained v.s. the number of
    #   components we retain), save the figure in file ``pov.jpg''
    #
    #   4. report how many dimensions are need to preserve 0.9 POV, describe
    #   your answers and your undestanding of the results in the plain text
    #   file ``description2.txt''
    #
    #   5. remember to submit file ``eigenimages.jpg'', ``pov.jpg'',
    #   ``description2.txt'' and ``ex2.py''.
    #
    # YOUR CODE HERE!
    ####################################################################
    eigenvalues, eigenVectors = pca(X)
    plot_eigen_image(eigenVectors)
    plot_pov(eigenvalues)

if __name__ == '__main__':
    main()
