from __future__ import print_function
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from math import sqrt, isclose

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn import metrics

def create_dataset():
    # Generate sample points
    centers = [[3,5], [5,1], [8,2], [6,8], [9,7]]
    X, y = make_blobs(n_samples=1000,centers=centers,cluster_std=[0.5, 0.5, 1, 1, 1],random_state=3320)
    ####################################################################
    # you need to
    #   1. Plot the data points in a scatter plot.
    #   2. Use color to represents the clusters.
    #
    # YOUR CODE HERE!
    ####################################################################
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title("show original data cluster")
    plt.show()


    return [X, y]

def initCentroids(f1max, f1min, f2max, f2min):

    centroidf1 = np.random.uniform(f1min,f1max)
    centroidf2 = np.random.uniform(f2min,f2max)
    centriod = [centroidf1,centroidf2]

    return centriod

def getEuclidDist(centroid, data):
    return sqrt(sum((centroid - data) ** 2))

def compareCentroid(prev_cen,current_cen):

    sum_prevX = 0.0
    sum_prevY = 0.0
    sum_currentX = 0.0
    sum_currentY = 0.0
    for x in range(len(prev_cen)):
        sum_prevX = sum_prevX + prev_cen[x][0]
        sum_prevY = sum_prevY + prev_cen[x][1]
    for x in range(len(current_cen)):
        sum_currentX = sum_currentX + current_cen[x][0]
        sum_currentY = sum_currentY + current_cen[x][1]
    if isclose(sum_prevX,sum_currentX) and isclose(sum_prevY,sum_currentY):
        return False
    else:
        return True

def my_clustering(X, y, n_clusters):
    # =======================================
    # you need to
    #   1. Implement the k-means by yourself
    #   and cluster samples into n_clusters clusters using your own k-means
    #
    #   2. Print out all cluster centers and sizes.
    #
    #   3. Plot all clusters formed,
    #   and use different colors to represent clusters defined by k-means.
    #   Draw a marker (e.g., a circle or the cluster id) at each cluster center.
    #
    #   4. Return scores like this: return [score, score, score, score]
    #
    # YOUR CODE HERE!
    ####################################################################

    ### 1. k-mean:

    # this is for generating random centriods
    f1max = max(X[:,0])
    f1min = min(X[:,0])
    f2max = max(X[:,1])
    f2min = min(X[:,1])
    centriods = []
    for x in range(n_clusters):
        centriods.append(initCentroids(f1max,f1min,f2max,f2min))

    # the logic of Kmeans
    continueIterate = True
    previousCentroids = []
    while continueIterate:
        ### assign points to cluster
        clustered_data_for_updating_centroids = []
        predict_label = []
        for k in range(n_clusters):
            clustered_data_for_updating_centroids.append([])
        for x in range(len(X[:,0])):
            dist = []
            for i in range(n_clusters):
                dist.append(getEuclidDist(centriods[i],X[x]))
            clustered_data_for_updating_centroids[dist.index(min(dist))].append(X[x])
            predict_label.append(dist.index(min(dist)))

        ### compare whether the cluster has been converged


        if len(previousCentroids) == 0 or compareCentroid(previousCentroids, centriods):
            previousCentroids = centriods
            ### update centriod
            for o in range(n_clusters):
                x_cen = 0.0
                y_cen = 0.0
                if len(clustered_data_for_updating_centroids[o]) != 0:
                    for m in range(len(clustered_data_for_updating_centroids[o])):
                        x_cen = x_cen + clustered_data_for_updating_centroids[o][m][0]
                        y_cen = y_cen + clustered_data_for_updating_centroids[o][m][1]
                    centriods[o][0] = x_cen/(len(clustered_data_for_updating_centroids[o]))
                    centriods[o][1] = y_cen/(len(clustered_data_for_updating_centroids[o]))
        else:
            ### converged, exit the loop
            continueIterate = False


    ari = metrics.adjusted_rand_score(y, predict_label)
    mri = metrics.mutual_info_score(y, predict_label)
    v_measure = metrics.v_measure_score(y, predict_label)
    sil = metrics.silhouette_score(X, predict_label, metric='euclidean')

    for x in range(len(centriods)):
        print("centriods"+str(x+1)+":",centriods[x])
        print("centriods size",len(clustered_data_for_updating_centroids[x]))

    centriods = np.array(centriods)
    plt.scatter(X[:, 0], X[:, 1], c=predict_label)
    plt.scatter(centriods[:,0],centriods[:,1], marker="x")
    plt.title("Number of Cluster: "+ str(n_clusters))
    plt.show()


    #print(x_cen)
    #print(y_cen)
    #print(clustered_data_for_updating_centroids[0])
    #print(clustered_data_for_updating_centroids[1])

    ### verify with skleran
    #kmeans = KMeans(n_clusters)
    #kmeans.fit(X)

    #print("ari",metrics.adjusted_rand_score(y, predict_label))
    #print("ari skleran",metrics.adjusted_rand_score(y, kmeans.labels_))

    #print("mri",metrics.mutual_info_score(y, predict_label))
    #print("mri sklearn",metrics.mutual_info_score(y, kmeans.labels_))


    return [ari,mri,v_measure,sil]  # You won't need this line when you are done

def main():
    X, y = create_dataset()
    range_n_clusters = [2, 3, 4, 5, 6, 7]
    #range_n_clusters = [3]
    ari_score = [None] * len(range_n_clusters)
    mri_score = [None] * len(range_n_clusters)
    v_measure_score = [None] * len(range_n_clusters)
    silhouette_avg = [None] * len(range_n_clusters)

    for n_clusters in range_n_clusters:
        i = n_clusters - range_n_clusters[0]
        print("Number of clusters is: ", n_clusters)
        # Implement the k-means by yourself in the function my_clustering
        [ari_score[i], mri_score[i], v_measure_score[i], silhouette_avg[i]] = my_clustering(X, y, n_clusters)
        print('The ARI score is: ', ari_score[i])
        print('The MRI score is: ', mri_score[i])
        print('The v-measure score is: ', v_measure_score[i])
        print('The average silhouette score is: ', silhouette_avg[i])

    ####################################################################
    # Plot scores of all four evaluation metrics as functions of n_clusters in a single figure.
    #
    # YOUR CODE HERE!
    ####################################################################

    x_axis_comp = range_n_clusters
    plot = plt.figure()
    plt.plot(x_axis_comp, ari_score, 'xr-', label = "ari")
    plt.plot(x_axis_comp, mri_score, 'xb-', label = "mri")
    plt.plot(x_axis_comp, v_measure_score, 'xg-', label = "v_measure")
    plt.plot(x_axis_comp, silhouette_avg, 'xy-', label = "silhouette")
    plt.legend(loc='best')
    plt.xlabel('number of cluster:')
    plt.ylabel('evaluation metrics:')
    plt.show()



if __name__ == '__main__':
    main()

