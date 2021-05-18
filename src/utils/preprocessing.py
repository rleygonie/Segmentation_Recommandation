from pyspark.ml.feature import StringIndexer
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import rand
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StringIndexer


def index_customers_data(data):
    stringIndexer = StringIndexer(inputCol="Gender", outputCol="GenderIndexed")
    model = stringIndexer.fit(data)
    indexed = model.transform(data)
    indexed = indexed.drop('Gender','CustomerID')
    indexed = indexed.select('Age','Annual Income (k$)','GenderIndexed','Spending Score (1-100)')
    return indexed
    
    
    
    
def vectoriz_customers_data(indexed_data):   
    vector_col = 'features'
    assembler = VectorAssembler(inputCols=indexed_data.columns, outputCol='features')
    df_vector = assembler.transform(indexed_data).select(vector_col)
    return df_vector


def correlation_matrix(df):
    matrix = Correlation.corr(df, 'features')
    return matrix.collect()[0]["pearson({})".format('features')].values




def create_Df_movie(ratings_movies):
    df = ratings_movies.withColumn('userId', (rand()*200+1))
    df = df.withColumn('userId', df['userId'].cast(IntegerType()).alias('userId'))
    df = df.withColumn('movieId', df['movieId'].cast(IntegerType()).alias('movieId'))
    return df

