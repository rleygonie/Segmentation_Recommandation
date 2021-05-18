from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.sql.functions import col
from pyspark.sql.functions import abs

import pandas as pd



def find_best_kmeans_k(data,k_max):
    silh=[]
    k=[]
    for i in range(2,k_max):
        kmeans = KMeans().setK(i).setSeed(1)
        model = kmeans.fit(data)
        predictions = model.transform(data)
        evaluator = ClusteringEvaluator()
        silhouette = evaluator.evaluate(predictions)
        print("k=",i)
        k.append(i)
        print("Silhouette with squared euclidean distance = " + str(silhouette))
        silh.append(silhouette)
        centers = model.clusterCenters()
        print("Cluster Centers: ")
        for center in centers:
            print(center)
            print()
    return [k[silh.index(max(silh))],max(silh)]




def find_best_BisectingKmeans_k(data,k_max):
    silh=[]
    k=[]
    for i in range(2,k_max):
        bkm = BisectingKMeans().setK(i).setSeed(1)
        model = bkm.fit(data)
        predictions = model.transform(data)
        evaluator = ClusteringEvaluator()
        silhouette = evaluator.evaluate(predictions)
        print("k=",i)
        k.append(i)
        print("Silhouette with squared euclidean distance = " + str(silhouette))
        silh.append(silhouette)
        print("Cluster Centers: ")
        centers = model.clusterCenters()
        for center in centers:
            print(center)
    return [k[silh.index(max(silh))],max(silh)]




def compare_clustering_models(bestKmeans,bestBKmeans):
    L=[bestKmeans[1],bestBKmeans[1]]
    if L.index(max(L)):
        print("Silhouette = ",bestBKmeans[1]," of BistectingKmeans with k = ",bestBKmeans[0],"is better than silhouette = ",bestKmeans[1]," of Kmeans with k = ",bestKmeans[0])
        return bestBKmeans[0]
    else :
        print("Silhouette = ",bestKmeans[1]," of Kmeans with k = ",bestKmeans[0],"is better than silhouette = ",bestBKmeans[1]," of BisectingKmeans with k = ",bestBKmeans[0])
        return bestKmeans[0]
    
    
    
    
