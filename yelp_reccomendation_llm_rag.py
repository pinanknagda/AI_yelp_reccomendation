import os

os.environ['JAVA_HOME'] = 'C:\\Program Files\\Java\\jdk-17'
os.environ['SPARK_HOME'] = 'C:\\spark-3.5.4'
import findspark

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import lower
from pyspark.sql import functions as F
from pyspark.sql.functions import to_timestamp,col

from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

findspark.init()

conf = SparkConf() \
    .setAppName("My Spark App") \
    .setMaster("local[*]") \
    .set("spark.driver.host", "192.168.1.245") \
    .set("spark.driver.bindAddress", "192.168.1.245") \
    .set("spark.driver.memory", "4g") \
    .set("spark.executor.memory", "4g") \
    .set("spark.sql.shuffle.partitions", "4000") \
    .set("spark.default.parallelism", "4000") \
    .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .set("spark.kryoserializer.buffer.max", "2000m")

sc = SparkContext.getOrCreate(conf=conf)

spark = SparkSession.builder.config(conf = conf).getOrCreate()

sc.setLogLevel("DEBUG")  # Set log level to DEBUG



reviews_df = spark.read.json('yelp_dataset\yelp_academic_dataset_review.json').withColumn('Date',to_timestamp(col('date'),'yyyy-MM-dd HH:mm:ss'))
business_df = spark.read.json('yelp_dataset\yelp_academic_dataset_business.json')

business_df = business_df.drop('attributes','hours')

business_df = business_df.where(lower(col('Categories')).rlike('.*restaurant.*'))
business_id_df = business_df.select(col('business_id')).filter(col('is_open') == 1)



def iqr_outliers(dataframe,column,factor=1.5):
    quartiles = dataframe.approxQuantile(column,[0.25,0.75],0.25)
    q1 , q3 = quartiles[0],quartiles[1]
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    dataframe = dataframe.filter((col(column) >= lower_bound) & (col(column) <= upper_bound))
    return dataframe

business_df = iqr_outliers(business_df.filter(col('is_open') == 1),'review_count')

filter_business_df = business_df.where((business_df.city == "Boise") & (business_df.state == "ID"))
business_review_df = filter_business_df.join(reviews_df,reviews_df.business_id ==  filter_business_df.business_id,"left")

filtered_df = business_review_df.select(col('review_id'), col('text'))

openai_api_key = os.environ["OPENAI_API_KEY"]

# Initialize embeddings outside the function
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Initialize vector store outside the function
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)

# Define the process_partition function
def process_partition(partition):
    for row in partition:
        vector_store.add(ids=[str(row.review_id)], documents=[str(row.text)])

# Filter the DataFrame to only include necessary columns
filtered_df = business_review_df.select(col('review_id'), col('text'))

# Apply the process_partition function to each partition
filtered_df.foreachPartition(process_partition)

spark.stop()

relevant_docs_retriever = vector_store.as_retriever(search_type = "similarity", k = 5)

prompt = ChatPromptTemplate.from_template("You are an Restaurent reccomendation assistant giving 5 reccomendations based on users query. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {user_query} \nContext: {context} \nAnswer:")


model = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4o-mini", temperature=0)

# Setup the chain
rag_chain = (
    {"context": relevant_docs_retriever , "user_query": RunnablePassthrough()}
    | prompt
    | model
)


# Initialize query
user_query = "i love thai food within 5 miles of Pilladelphia central market"

# Invoke the query
answer = rag_chain.invoke(user_query).content
print(answer)