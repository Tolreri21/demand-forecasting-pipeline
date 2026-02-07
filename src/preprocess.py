# Import required libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql.functions import lit, col, dayofmonth, month, year,  to_date, to_timestamp, weekofyear, dayofweek
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("SalesForecast")\
    .master("local[*]")\
    .getOrCreate()


raw_data = spark.read.csv("../data/raw/Online Retail.csv", header=True, inferSchema=True)
print(raw_data.count())

raw_data = raw_data.drop("Year" , "Month" ,"Week" , "Day" , "DayOfWeek")

raw_data= raw_data.withColumn("InvoiceDate" , to_date(to_timestamp(col("InvoiceDate"))))
raw_data= raw_data.withColumn("Year" , year(col("InvoiceDate")))
raw_data= raw_data.withColumn("Month" , month(col("InvoiceDate")))
raw_data= raw_data.withColumn("Week" , weekofyear(col("InvoiceDate")))
raw_data= raw_data.withColumn("Day" , dayofmonth(col("InvoiceDate")))
raw_data= raw_data.withColumn("DayOfWeek" , dayofweek(col("InvoiceDate")))

raw_data_aggted = raw_data.groupBy("InvoiceDate" , "Year", "Month" , "Week" , "Day" , "DayOfWeek" , "StockCode" , "Country").agg({"Quantity" : "sum" , "UnitPrice" : "avg" })
raw_data_aggted = raw_data_aggted.withColumnRenamed("sum(Quantity)" , "Quantity")
print(raw_data_aggted.show(5))

date_to_split = to_date(lit("2011-09-25"))

raw_data_aggted_train = raw_data_aggted.filter(col("InvoiceDate")  <= date_to_split)
raw_data_aggted_test = raw_data_aggted.filter(col("InvoiceDate")  > date_to_split)



print(f"amount of rows in train : {raw_data_aggted_train.count()} amount of rows in test : {raw_data_aggted_test.count()}")

country_indexer = StringIndexer(inputCol="Country", outputCol="CountryIndex")
stock_code = StringIndexer(inputCol="StockCode", outputCol="StockCodeIndex")



country_indexer_model = country_indexer.fit(raw_data_aggted_train)
raw_data_aggted_test = country_indexer_model.transform(raw_data_aggted_test)
raw_data_aggted_train = country_indexer_model.transform(raw_data_aggted_train)


stock_code_indexer_model = stock_code.fit(raw_data_aggted_train)
raw_data_aggted_test = stock_code_indexer_model.transform(raw_data_aggted_test)
raw_data_aggted_train = stock_code_indexer_model.transform(raw_data_aggted_train)


X_train = raw_data_aggted_train.select("CountryIndex", "StockCodeIndex", "Month", "Year","DayOfWeek", "Day", "Week")
X_test = raw_data_aggted_test.select("CountryIndex", "StockCodeIndex", "Month", "Year", "DayOfWeek", "Day", "Week")
y_train = raw_data_aggted_train.select("Quantity")
y_test = raw_data_aggted_test.select("Quantity")

# Saving data
X_train.write.mode("overwrite").parquet(r"C:\Users\TOLK\PycharmProjects\demand_forecasting\data\processed/X_train.parquet")
X_test.write.mode("overwrite").parquet(r"C:\Users\TOLK\PycharmProjects\demand_forecasting\data\processed/X_test.parquet")

y_train.write.mode("overwrite").parquet(r"C:\Users\TOLK\PycharmProjects\demand_forecasting\data\processed/y_train.parquet")
y_test.write.mode("overwrite").parquet(r"C:\Users\TOLK\PycharmProjects\demand_forecasting\data\processed/y_test.parquet")



























spark.stop()