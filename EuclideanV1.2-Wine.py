
from pyclustering.cluster.kmeans import kmeans
from sklearn import preprocessing
from sklearn.cluster import KMeans
from numpy import genfromtxt
from scipy.spatial import distance

import numpy as np

r, c = 178, 3
results = [[0 for x in range(c)] for y in range(r)]
# print(len(results))
k = genfromtxt('wine_data.csv', delimiter=',')
k = np.ma.compress_cols(np.ma.masked_invalid(k))
l = k
k = k[:,1:len(k[0])]

k_norm = preprocessing.scale(k)

kmeans = KMeans(n_clusters=3,random_state=0).fit(k_norm)
print(kmeans.labels_)

dist_list = list()

count =1
for row in k_norm:
    dist_list_row = list()
    for row1 in k_norm:
        val = distance.euclidean(row,row1)
        dist_list_row.append(val)
    dist_list.append(dist_list_row)
    # k_means = KMeans(n_clusters=2,random_state=0).fit(dist_list_row)
    # print("Distance for row ")
    # print(count)
    # count = count + 1
    # print(dist_list_row)
    # print(kmeans)
    # optics_instance.process()
    #
    # # Creating a 2d array of Clusters which is not uniform.
    # #clusters[cluster #][cluster elements label]
    # clusters = optics_instance.get_clusters()
    #
    # print("Clusters")
    # print(clusters)
    #
    # noise = optics_instance.get_noise()
    # print("Noise")
    # print(noise)

    # clusterCount = 1
    # for k in clusters:
    #    for temp in k: #For each element in the cluster k
    #        if clusterCount <= c:
    #            results[temp][clusterCount-1]  = results[temp][clusterCount-1] + 1
    #
    #        else:
    #            results[temp][c-1] = results[temp][c-1] + 1
    #            #continue
    #
    #    clusterCount = clusterCount + 1
    #
    # print("Number of clusters")
    #
    # print(len(clusters))
#End of outer for loop
print("--------------Clustering Results-------------")


# output = open('results.csv', 'w')


# print(len(results))
output = open('results.csv', 'w')
c = 0
tempRowNum = 1
for row in results:
    # print("row: " + str(tempRowNum) + " ****has count of cluster 1: " + str(row[0]) + " ****has count of cluster 2: " + str(row[1])
    #       + " ****has count of cluster 3: " + str(row[2]))

    output.write(str(tempRowNum))
    output.write(str(","))
    output.write(str(row.index(max(row)) + 1))
    output.write(str("\n"))
    tempRowNum = tempRowNum + 1

output.close()

print("--------------Clustering Results Writen to results.csv-------------")