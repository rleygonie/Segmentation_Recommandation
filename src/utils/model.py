
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.sql.functions import col
from pyspark.sql.functions import abs

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

import pandas as pd



def kmeans_best_clustering(data,k):
    kmeans = KMeans().setK(k).setSeed(1)
    kmeans.setMaxIter(300)
    model = kmeans.fit(data)
    model.setPredictionCol("Cluster")
    predictions = model.transform(data)
    return predictions


def create_df_from_predictions(predictions):
    pandasDF = predictions.toPandas()
    gender=[]
    age=[]
    salaire=[]
    Spending_Score=[]
    predi=[]
    for i in range(len(pandasDF)):
        age.append(pandasDF["features"][i][0])
        salaire.append(pandasDF["features"][i][1])
        gender.append(pandasDF["features"][i][2])
        Spending_Score.append(pandasDF["features"][i][3])
        predi.append(pandasDF["Cluster"][i])
        data = {'Gender':  gender,
            'Age':age,
            'Annual Income (k$)': salaire,
            'Spending Score (1-100)':Spending_Score,
            'Cluster': predi
            }

    df = pd.DataFrame (data, columns=["Age","Annual Income (k$)","Gender","Spending Score (1-100)","Cluster"])
    return df


def rf_train(train_data):
    rf = RandomForestClassifier(labelCol="Cluster", featuresCol="features")
    model_tree = rf.fit(train_data)
    return model_tree

def rf_eval_test(model, test_data):
    predictions = model.transform(test_data)
    predictions.show()
    evaluator = MulticlassClassificationEvaluator(labelCol="Cluster", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    return accuracy
    




def Dt_train(train_data):
    dt = DecisionTreeClassifier(labelCol="Cluster", featuresCol="features")
    model = dt.fit(train_data)
    return model

def Dt_eval_test(model,test_data):
    predictions = model.transform(test_data)
    predictions.show()
    evaluator = MulticlassClassificationEvaluator(labelCol="Cluster", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    return accuracy


   
def ALS_train(train_data):
    model = ALS(userCol='userId', itemCol='movieId', ratingCol='rating').fit(train_data)
    return model


def ALS_eval_test(model,test_data):
    predictions = model.transform(test_data)
    predictions.show()
    predictions = predictions.na.drop()
    evaluator = RegressionEvaluator(metricName="rmse",labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print('rmse =',rmse)
    return predictions



def recommandation_movies_for_user(model,test,userId,movies):
    recommendation = model.recommendForItemSubset(test,numUsers=userId)
    df2 = recommendation.join(movies, movies.id == recommendation.movieId).select(recommendation["*"],movies["title"])
    df2.show(truncate=False)
    
    
