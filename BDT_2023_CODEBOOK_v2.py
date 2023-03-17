# Databricks notebook source
# DBTITLE 1, Big Data Tools Group Project : Members - Priya Yadav, Ashval Vaidya, Vasu Singh


# COMMAND ----------

#Load functions
from pyspark.sql.functions import *

# COMMAND ----------

#There are five training data sources:
"""
  products: information about the products
  orders: information about the orders 
  order_items : information about the items ordered
  order_payments : information about the payments
  order_reviews : information about the reviews of the orders
"""

# COMMAND ----------

# DBTITLE 1,ORDER_REVIEW TABLE
"""
Metadata description: 
  •	review_id: Unique identifier for a review
  •	order_id: Order unique identifier
  •	review_score: Score from 1 to 5 given by the customer on a customer satisfaction survey
  •	review_creation_date: Date at which the customer satisfaction survey was sent to the customer.
  •	review_answer_timestamp: Timestamp at which the customer answered to the customer satisfaction survey.
 
"""

#Read table: order_reviews
order_reviews=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load("/FileStore/tables/order_reviews.csv")

#print order_reviews
display(order_reviews)

#print schema of order_reviews to check the type of the column
order_reviews.printSchema()

# COMMAND ----------

#Convert the data type of the "review_answer_timestamp" from string to timestamp
order_reviews= order_reviews.select("*", col("review_answer_timestamp").cast("timestamp").alias("review_answer_time"))

#Drop the string column "review_answer_timestamp"
order_reviews = order_reviews.drop("review_answer_timestamp")

#print schema of order_reviews to check the type of the column
order_reviews.printSchema()


# COMMAND ----------

#inspect the total number of rows.
order_reviews.count()

#inspect the unique number of rows.
order_reviews.distinct().count()

# Count the number of null values in each column
order_reviews.select([sum(col(c).isNull().cast("int")).alias(c) for c in order_reviews.columns]).show(3)

#describe the table
order_reviews.describe().show(3)

# COMMAND ----------

#Replacing NULL values in "review_answer_time" with "review_creation_date" + 1 extra day for reviewing.
from pyspark.sql import functions as F
order_reviews = order_reviews.withColumn("review_answer_time", 
                   F.when(F.isnull(order_reviews["review_answer_time"]), 
                          F.date_add(order_reviews["review_creation_date"], 1)).otherwise(order_reviews["review_answer_time"]))
order_reviews.show()

#Inspect for NULL values again
# Count the number of null values in each column
order_reviews.select([sum(col(c).isNull().cast("int")).alias(c) for c in order_reviews.columns]).show(3)


# COMMAND ----------

#order_reviews table

#There are certain observations where a single order_id is linked with multiple review_id.

#In order to deal with it, we are only keeping the most recent review_id based on the review_answer_timestamp (time).



#Total number of order_id where review given were 1 or more than one.
or_count = order_reviews.groupBy("order_id").agg(count("review_id").alias("review_count"))
or_count = or_count.groupBy("review_count").agg(count("order_id").alias("order_count"))
or_count.show(3)

#count the reviews id generated on each order id to check that whether each order id have only one review id
order_reviews1= order_reviews.groupBy("order_id").agg(count("review_id").alias("review_count"))
order_reviews1.show(3)

#join with order_reviews
order_reviews= order_reviews.join(order_reviews1, "order_id", "left")
display(order_reviews)



# COMMAND ----------

#Creating two datasets, for order_id where only single reivew was given and we will keep that dataset as it as. 

#Creating another dataset where we have multiple reviews per order_id, here we will only select the latest review.


#create order_first that have only 1 reviews generated on unique order_id
order_first= order_reviews.where(col("review_count") == 1)
order_first.show(2)
order_first.count()

#create order_second that contains duplicate order id on which more than 1 reviews has been generated 
order_second = order_reviews.where(col("review_count") != 1)
order_second.show(2)
order_second.count()

#sorting dataFrame order_second for review_answer_time descending and selecting the latest review answer.
order_second= order_second.orderBy(col("order_id").asc()).orderBy(col("review_answer_time").asc())

from pyspark.sql import Window

windowSpec = Window.partitionBy("order_id").orderBy("review_answer_time").rangeBetween(Window.unboundedPreceding, Window.unboundedFollowing)

unique_order_reviews= order_second.select("*", max("review_answer_time").over(windowSpec).alias("Max_Timestamp")).where(col("review_answer_time") == col("Max_Timestamp")).drop("Max_Timestamp")
unique_order_reviews.show(3)
unique_order_reviews.count()

# COMMAND ----------

#now merge the order_first and unique_order_reviews to create orders_reviews that have unique order_id with one reviews generated on it
order_reviews= order_first.union(unique_order_reviews)
order_reviews.show(3)
order_reviews.count()

# COMMAND ----------

# DBTITLE 1,Creating Label from the Review_score column
# "Positive": where review_score is greater than or equal to 4
# "Negative": where review_score is less than 4
order_reviews = order_reviews.withColumn("Review_Class", when(col("review_score") < 4, "Negative").otherwise("Positive"))

order_reviews = order_reviews.withColumn("label", when(order_reviews["Review_Class"] == "Positive", lit(1)).otherwise(lit(0)))

# Drop the original "review_class" column
order_reviews = order_reviews.drop("review_class")

display(order_reviews)

# COMMAND ----------

#create a column DaysTakenForReview to find the number of days taken to submit the review after the review was generated for an orderid.
order_reviews =  order_reviews.withColumn("DaysTakenForReview",datediff("review_answer_time","review_creation_date"))


#------------------------------------------


#Create a new column "time_of_day" that represents the hour of the day at which the customer answered to the customer satisfaction survey. 
# "review_answer_time" between 06:00:00 and 11:59:59 as "morning"
# "review_answer_time" between 12:00:00 and 17:59:59 as "afternoon"
# "review_answer_time" between 18:00:00 and 19:59:59 as "evening"
# "review_answer_time" between 20:00:00 and 5:59:59 as "night"

order_reviews = order_reviews.withColumn("review_time_of_day", 
                        when(hour(order_reviews["review_answer_time"]).between(6, 11), "morning")
                        .when(hour(order_reviews["review_answer_time"]).between(12, 17), "afternoon")
                        .when(hour(order_reviews["review_answer_time"]).between(18, 19), "evening")
                        .otherwise("night"))


#--------------------------


#Create a new column "weekend_or_weekday" that represents the day of the week during which the customer answered to the customer satisfaction survey. 
# "review_answer_time" < 5 as "weekday"
# "review_answer_time" > 5 as "weekend"

from pyspark.sql.functions import when, date_format, dayofweek

order_reviews = order_reviews.withColumn("review_weekend_or_weekday", 
                        when(dayofweek(order_reviews["review_answer_time"]) <= 5, "weekday")
                        .otherwise("weekend"))

#--------------------------


# created new column for day of the week the review was given.
order_reviews = order_reviews.withColumn("day_of_the_week", 
                      when(dayofweek(order_reviews["review_answer_time"]) == 1, "Monday")
                      .when(dayofweek(order_reviews["review_answer_time"]) == 2, "Tuesday")
                      .when(dayofweek(order_reviews["review_answer_time"]) == 3, "Wednesday")
                      .when(dayofweek(order_reviews["review_answer_time"]) == 4, "Thursday")
                      .when(dayofweek(order_reviews["review_answer_time"]) == 5, "Friday")
                      .when(dayofweek(order_reviews["review_answer_time"]) == 6, "Saturday")
                      .otherwise("Sunday"))

#--------------------------

#Creating a new feature for month of the year in which the review was given.
from pyspark.sql.functions import month, when

order_reviews = order_reviews.withColumn("month", 
                      when(month(order_reviews["review_answer_time"]) == 1, "January")
                      .when(month(order_reviews["review_answer_time"]) == 2, "February")
                      .when(month(order_reviews["review_answer_time"]) == 3, "March")
                      .when(month(order_reviews["review_answer_time"]) == 4, "April")
                      .when(month(order_reviews["review_answer_time"]) == 5, "May")
                      .when(month(order_reviews["review_answer_time"]) == 6, "June")
                      .when(month(order_reviews["review_answer_time"]) == 7, "July")
                      .when(month(order_reviews["review_answer_time"]) == 8, "August")
                      .when(month(order_reviews["review_answer_time"]) == 9, "September")
                      .when(month(order_reviews["review_answer_time"]) == 10, "October")
                      .when(month(order_reviews["review_answer_time"]) == 11, "November")
                      .otherwise("December"))


#Dropping un-necessary or intermediate columns (review_count) which we to get this dataFrame
order_reviews = order_reviews.drop("review_count")

display(order_reviews)

# COMMAND ----------

#metadata of final_orders_reviews
order_reviews.printSchema()

# COMMAND ----------

# DBTITLE 1,(ORDER_ITEMS & PRODUCTS) TABLES
"""
Metadata description: 
  •	order_id: Order unique identifier
  •	order_item_id: Sequential number identifying the order of the ordered items. A customer can order multiple items per order
  •	product_id: Product unique identifier
  •	price: user Item price in euro (excl. VAT)
  •	shipping_cost: Cost for shipping the item to the customer in euro (excl. VAT)
 
"""

# COMMAND ----------

#Read table: orders_item
order_items=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load("/FileStore/tables/order_items.csv")

#print orders_item
order_items.show(3)

#print metadata of orders_item to check the type of the column
order_items.printSchema()

# COMMAND ----------

#Remove duplicates
order_items = order_items.dropDuplicates()

#total number of observation(rows) in orders_item
print("count",order_items.count())

#Inspect the table
order_items.describe().show()

# COMMAND ----------

"""
Metadata description: 
product_id: Product unique identifier
product_name_length: Number of characters in the product name
product_description_length: Number of characters in the product description
product_photos_qty: Number of photos included in the product description
product_weight_g: Product weight (in grams)
product_length_cm: Product dimensions - length (in centimeters)
product_height_cm: Product dimensions - height (in centimeters)
product_width_cm: Product dimensions - width (in centimeters)
product_category_name: Product category name
  
"""

# COMMAND ----------

#Read table: product
products = spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.option("escape","\"")\
.load("/FileStore/tables/products.csv")

#inspect the table
products.show(3)

#check the datatypes
order_items.printSchema()

#describe the products table
display(products.describe())

#Count the total no of observations
print("count", products.count())

#inspect the distinct number of rows.
print("distinct count",products.distinct().count())

# COMMAND ----------

# DBTITLE 1,Joining the two tables (order_items & products)
#joining the two tables based on common column called "product_id"
order_items_products = order_items.join(products, on="product_id", how="left")

#inspect the table
display(order_items_products)

#check the summary of the new table
display(order_items_products.describe())

#check the count
print("count",order_items_products.count())

# COMMAND ----------

#Checking for the NULL Columns
# Count the number of null values in each column
display(order_items_products.select([sum(col(c).isNull().cast("int")).alias(c) for c in order_items_products.columns]))

#inspect the NULL values 
display(order_items_products.filter(col("product_name_lenght").isNull()))

#-----------------------

# replacing NULL values in "product_name_length", "product_description_lenght", "product_photos_qty","product_weight_g","product_length_cm","product_height_cm"
# "product_width_cm","product_category_name"

# compute the mean of each column
col_mean = {col: order_items_products.filter(order_items_products[col].isNotNull()).groupBy(order_items_products[col]).count().agg(mean(order_items_products[col])).first()[0] for col in ["product_name_lenght","product_description_lenght","product_photos_qty","product_weight_g","product_length_cm","product_height_cm","product_width_cm"]}

# # replace NULL values with the mean
order_items_products = order_items_products.na.fill(col_mean)

#-----------------------


# Replacing NULL values of "product_category_name" with "Others"
m = order_items_products.groupBy(order_items_products.product_category_name).agg(count(order_items_products.product_category_name).alias("count")).orderBy(desc("count")).first()[0]

order_items_products = order_items_products.na.fill(value=m,subset=["product_category_name"])

# COMMAND ----------

#Checking the NULL Values once again
# Count the number of null values in each column
display(order_items_products.select([sum(col(c).isNull().cast("int")).alias(c) for c in order_items_products.columns]))

# COMMAND ----------

# removing the duplicates
order_items_products = order_items_products.dropDuplicates()

#check the count
order_items_products.count()


#to check if there is any erroneous values for column price and shipping_cost
from pyspark.sql.functions import *
od= order_items_products.where((col("price")<=0) & (col("shipping_cost")<= 0)).select("order_id", "price", "shipping_cost")
display(od)

# COMMAND ----------

#check the count
print("count",order_items_products.count())

#unit of analysis is order_id so check how many unique orders there are
print("distinct Count", order_items_products.select("order_id").distinct().count())

# COMMAND ----------

# Performing aggregations on the "order_items_products" and making it unique "order_id". 
# unique "order_id" is the level of granularity we are keeping

# creating below small dataframe to get the desired result and later merging them to get the final output dataframe

#-------------------------

# 1) 
#count the total number of items bought by the customer per order id
total_items= order_items_products.groupBy("order_id").agg(count("order_item_id").alias("total_items"))
display(total_items)

total_items.count()

display(total_items)


#-------------------------

# 2) 
#total price, total shipping cost, average price and average shipping cost of items paid by the customer per order id
total_cost= order_items_products.groupBy("order_id").\
agg(round(sum("price"), 2).alias("total_price"), \
    round(sum("shipping_cost"), 2).alias("total_shipping_cost"), \
    round(avg("price"), 2).alias("average_price"), \
    round(avg("shipping_cost"), 2).alias("average_shipping_cost"),\
    round(avg("product_name_lenght"), 2).alias("average_product_name_lenght"),\
    round(avg("product_description_lenght"), 2).alias("average_product_description_lenght"), \
    round(avg("product_photos_qty"), 2).alias("average_product_photos_qty"),\
    round(avg("product_weight_g"), 2).alias("average_product_weight_g"))

#Calculating the total_amount = total_price + total_shipping_cost
total_cost = total_cost.withColumn("total_amount", expr("total_price + total_shipping_cost"))

display(total_cost)


#-------------------------

# 3) 
dummy_product_category_type = order_items_products.groupBy("order_id").pivot("product_category_name").agg(sum(when(col("price").isNotNull(), col("price")) + when(col("shipping_cost").isNotNull(), col("shipping_cost"))).alias("total_cost")).na.fill(0)

# rename dummy variablescolumn_list = dummy_payment_sequential.columns
column_list = dummy_product_category_type.columns
prefix = "total_amount_"
new_column_list = [prefix + s for s in column_list]
new_column_list

dummy_product_category_type = dummy_product_category_type.toDF(*new_column_list)

#renaming the Order_id column back 
dummy_product_category_type = dummy_product_category_type.withColumnRenamed("total_amount_order_id","order_id")

display(dummy_product_category_type)

# COMMAND ----------

# Merging all the above dataframes together

# which are - total_items + total_cost + dummy_product_category_type 
order_items_products_final = total_items.join(total_cost, on="order_id", how="left")

#merging the remaining table
order_items_products_final = order_items_products_final.join(dummy_product_category_type, on="order_id", how="left")

#verify the order_id count
print("order Id count", order_items_products_final.select("order_id").distinct().count())

#total rows count
print("count rows", order_items_products_final.count())

display(order_items_products_final)

# COMMAND ----------

# DBTITLE 1,ORDERS TABLE
#Read table: orders
orders = spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.option("escape","\"")\
.load("/FileStore/tables/orders.csv")

#Inspect all column datatypes
orders.printSchema()

print("count", orders.count())

display(orders)


# COMMAND ----------

# DBTITLE 1,INDEPENDENT VARIABLE TIMELINE CHECK
# selecting orders place between September 2020 and June 2022.
orders= orders.where((col("order_purchase_timestamp")>="2020-07-01 00:00:00") & (col("order_purchase_timestamp")<= "2022-06-30 23:59:59")).select("*")

print("count", orders.count())

#count the uniqhe number of order_id
print("distinct",orders.distinct().count())

# COMMAND ----------

# Count the number of null values in each column
display(orders.select([sum(col(c).isNull().cast("int")).alias(c) for c in orders.columns]))

# removing rows where all the columns were "NA" String.
orders = orders.filter(when(col("order_id") == "NA", None).otherwise(col("order_id")).isNotNull())

#check the row count after the removal of rows where order_id = "NA"
orders.count()

# COMMAND ----------

#Convert all date columns from "String" to "Date" format.
from pyspark.sql.types import *

orders= orders.select("*",
col("order_purchase_timestamp").cast("timestamp").alias("order_purchase_time"),
col("order_approved_at").cast("timestamp").alias("order_approved_at_time"),
col("order_delivered_carrier_date").cast("timestamp").alias("order_delivered_carrier_date_time"),
col("order_delivered_customer_date").cast("timestamp").alias("order_delivered_customer_date_time"),                      
col("order_estimated_delivery_date").cast("timestamp").alias("order_estimated_delivery_date_time"))

#As now the date columns type are converted from string to timestamp
orders.printSchema()

#check for NULL again
display(orders.select([sum(col(c).isNull().cast("int")).alias(c) for c in orders.columns]))

#drop the string date columns
orders= orders.drop("order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date")
orders.printSchema()

# COMMAND ----------

# # Remove all the rows where "order_delivered_customer_date_time" is missing.
# orders = orders.dropna(how="any")

# #check for NULL again
# display(orders.select([sum(col(c).isNull().cast("int")).alias(c) for c in orders.columns]))

# #check the count
# print("count", orders.count())

# COMMAND ----------

# Group the DataFrame by "order_id" and count the number of rows in each group
grouped_df = orders.groupBy("order_id").agg(count("*").alias("count"))

# Filter the groups where the count is greater than 1
filtered_df = grouped_df.filter(grouped_df["count"] > 1)

# Show the filtered DataFrame
filtered_df.show()

# COMMAND ----------

#From the above code, we could identify every observation is on "order_id"

#Find the number of days taken to submit the review after the review was generated for an orderid.
orders =  orders.withColumn("estimated_delivery_days",datediff("order_estimated_delivery_date_time","order_delivered_customer_date_time"))
orders =  orders.withColumn("order_to_approval_days",datediff("order_approved_at_time","order_purchase_time"))
orders =  orders.withColumn("actual_delivered_days",datediff("order_delivered_customer_date_time","order_purchase_time"))

#--------------------------------

#Creating a new feature(variable) for the difference between estimated delivery and delivery date
#"Early delivery" meant a POSITIVE value
#"On time delivery" meant a ZERO value
#"Late delivery" meant NEGATIVE value

#Creating dummy variable for whether order was "On Time", "Delayed" or "Early"
orders = orders.withColumn("DeliveryTimeStatus", when(col("estimated_delivery_days") > 0, "early")
                              .when(col("estimated_delivery_days") < 0, "delayed")
                              .otherwise("timely"))


#--------------------------------

#Create a new column "order_time_of_day" that represents the hour of the day at which the customer placed the order. 
# "order_purchase_time" between 06:00:00 and 11:59:59 as "morning"
# "order_purchase_time" between 12:00:00 and 17:59:59 as "afternoon"
# "order_purchase_time" between 18:00:00 and 19:59:59 as "evening"
# "order_purchase_time" between 20:00:00 and 5:59:59 as "night"

from pyspark.sql.functions import when, hour

orders = orders.withColumn("order_time_of_day", 
                        when(hour(orders["order_purchase_time"]).between(6, 11), "morning")
                        .when(hour(orders["order_purchase_time"]).between(12, 17), "afternoon")
                        .when(hour(orders["order_purchase_time"]).between(18, 19), "evening")
                        .otherwise("night"))

#--------------------------------

#Create a new column "weekend_or_weekday" that represents the day of the week during which the customer purchased the order. 
# "review_answer_time" < 5 as "weekday"
# "review_answer_time" > 5 as "weekend"

from pyspark.sql.functions import when, date_format, dayofweek

orders = orders.withColumn("order_purchase_weekend_or_weekday", 
                        when(dayofweek(orders["order_purchase_time"]) <= 5, "weekday")
                        .otherwise("weekend"))


#--------------------------------

#Create a new column "delivery_time_of_day" that represents the hour of the day at which the customer recieved the order. 
# "order_purchase_time" between 06:00:00 and 11:59:59 as "morning"
# "order_purchase_time" between 12:00:00 and 17:59:59 as "afternoon"
# "order_purchase_time" between 18:00:00 and 19:59:59 as "evening"
# "order_purchase_time" between 20:00:00 and 5:59:59 as "night"

from pyspark.sql.functions import when, hour

orders = orders.withColumn("delivery_time_of_day", 
                        when(hour(orders["order_delivered_customer_date_time"]).between(6, 11), "morning")
                        .when(hour(orders["order_delivered_customer_date_time"]).between(12, 17), "afternoon")
                        .when(hour(orders["order_delivered_customer_date_time"]).between(18, 19), "evening")
                        .otherwise("night"))

#--------------------------------

#Create a new column "weekend_or_weekday" that represents the day of the week on which the order was actually delivered. 
# "review_answer_time" < 5 as "weekday"
# "review_answer_time" > 5 as "weekend"

from pyspark.sql.functions import when, date_format, dayofweek

orders = orders.withColumn("order_delivered_weekend_or_weekday", 
                        when(dayofweek(orders["order_delivered_customer_date_time"]) <= 5, "weekday")
                        .otherwise("weekend"))

display(orders)

# COMMAND ----------

#converting columns to numeric.
int_columns = ["estimated_delivery_days", "order_to_approval_days", "actual_delivered_days"]

for col_name in int_columns:
    orders = orders.withColumn(col_name, col(col_name).cast("double"))

orders.printSchema()

# COMMAND ----------

# DBTITLE 1,ORDER_PAYMENT TABLE
"""
Metadata description: 
  •	order_id: Order unique identifier
  •	payment_sequential: Sequential number identifying the order of the payment types used. A customer may use several payment types for one order.
  •	payment_type: Method of payment chosen by the customer
  •	payment_installments: Number of installments chosen by the customer
  •	payment_value: Order value in euro
"""


#Read table: order_payment
order_payment=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load("/FileStore/tables/order_payments.csv")

#print order_payment
display(order_payment)

#print schema of order_payment to check the type of the column
order_payment.printSchema()

#total number of observation(rows) in order_payment
print("count", order_payment.count())

# COMMAND ----------

#Remove duplicates
order_payment = order_payment.dropDuplicates()

#Inspect the table
display(order_payment.describe())

#unit of analysis is order_id so check how many unique orders there are
order_payment.select("order_id").distinct().count()

# COMMAND ----------

#Unique values of column payment type
display(order_payment.select("payment_type").distinct().collect())

#Unique values of column payment type
display(order_payment.select("payment_installments").distinct().collect())

#Unique values of column payment_sequential
display(order_payment.select("payment_sequential").distinct().collect())


# COMMAND ----------

# Creating two intermedia DataFrames payment1 and payment2 to get the final payment dataframe.

# COMMAND ----------

#minimum and maximum payment_sequential and payment_installments choosen by the customer per order id
order_payment1= order_payment.groupBy("order_id").agg(min("payment_sequential").alias("min_payment_sequential"),                   max("payment_sequential").alias("max_payment_sequential"), min("payment_installments").alias("min_payment_installments"),                       max("payment_installments").alias("max_payment_installments"))
display(order_payment1)


# Filter the rows where col1 and col2 have the same values
same_value_rows4 = order_payment1.filter(order_payment1["min_payment_sequential"] == order_payment1["max_payment_sequential"])

# Count the number of rows
num_rows4 = same_value_rows4.count()

# Print the result
print("Number of rows where col1 and col2 have the same values:", num_rows4)

# Filter the rows where col1 and col2 have the same values
same_value_rows5 = order_payment1.filter(order_payment1["min_payment_installments"] == order_payment1["max_payment_installments"])

# Count the number of rows
num_rows5 = same_value_rows5.count()

# Print the result
print("Number of rows where col1 and col2 have the same values:", num_rows5)

order_payment1= order_payment1.drop("min_payment_sequential", "min_payment_installments")
display(order_payment1)

# COMMAND ----------

#total, average, min, max payment value per order id
order_payment2= order_payment.groupBy("order_id").agg(round(sum("payment_value"), 2).alias("total_payment_value"), round(avg("payment_value"),                           2).alias("average_payment_value"), min("payment_value").alias("min_payment_value"), max("payment_value").alias("max_payment_value"))
display(order_payment2)

order_payment2.count()

# Filter the rows where col1 and col2 have the same values
same_value_rows6 = order_payment2.filter(order_payment2["total_payment_value"] == order_payment2["average_payment_value"])

# Count the number of rows
num_rows6 = same_value_rows6.count()

# Print the result
print("Number of rows where col1 and col2 have the same values:", num_rows6)


# Filter the rows where col1 and col2 have the same values
same_value_rows7 = order_payment2.filter(order_payment2["total_payment_value"] == order_payment2["min_payment_value"])

# Count the number of rows
num_rows7 = same_value_rows7.count()

# Print the result
print("Number of rows where col1 and col2 have the same values:", num_rows7)

# Filter the rows where col1 and col2 have the same values
same_value_rows8 = order_payment2.filter(order_payment2["total_payment_value"] == order_payment2["max_payment_value"])

# Count the number of rows
num_rows8 = same_value_rows8.count()

# Print the result
print("Number of rows where col1 and col2 have the same values:", num_rows8)

order_payment2= order_payment2.drop("average_payment_value", "min_payment_value", "max_payment_value")

#join the order_payment1 and order_payment2 dataframe to create order_payment_agg
order_payment_agg= order_payment1.join(order_payment2, "order_id")
display(order_payment_agg)

order_payment_agg.count()



# COMMAND ----------

#join the order_payment1 and order_payment2 dataframe to create order_payment_agg
order_payment_agg= order_payment1.join(order_payment2, "order_id")
display(order_payment_agg)

print(order_payment_agg.count())

order_payment_agg.printSchema()

# COMMAND ----------

#Creating another intermediate dataframe for fetching payment_type corresponsing to each order_id.
#Payment sequential is unique for each order_id
order_payment_group = order_payment.select(["order_id","payment_type"])
order_payment_group.show(3)
order_payment_group = order_payment_group.groupby(col("order_id")).agg(first("payment_type").alias("payment_type"))
order_payment_group.show(5)
print(order_payment_group.count())

# COMMAND ----------

# merging it with the original dataframe 
order_payment_final= order_payment_group.join(order_payment_agg, "order_id", "left")

display(order_payment_final)

# COMMAND ----------

# DBTITLE 1,COUNT OF ALL DATATABLES USED ABOVE
#Checking the count of every table
print("order_reviews", order_reviews.count(), \
      "order_items_products_final",order_items_products_final.count(),\
      "orders",orders.count(),\
      "order_payment_final", order_payment_final.count())

# COMMAND ----------

# DBTITLE 1,BASETABLE and MERGING all data sets
#create basetable by joining tables in the following order:
# 1. order_reviews : 
# 2. order_items_products_final
# 3. orders
# 4. order_payment_final


basetable= order_reviews.join(orders, "order_id", "left")

basetable= basetable.join(order_payment_final, "order_id", "left")

basetable= basetable.join(order_items_products_final, "order_id", "left")

display(basetable)

# COMMAND ----------

basetable.count()

# COMMAND ----------

#Inspect for NULL values again
# Count the number of null values in each column

display(basetable.select([sum(col(c).isNull().cast("int")).alias(c) for c in basetable.columns]))


# COMMAND ----------

# treat the null values of the datediff columns with "99999", as null cannot be given to the model 
# and also to identify where we had the null values for these columns

basetable = basetable.na.fill(value=99999,subset=[
                                     "estimated_delivery_days",
                                     "order_to_approval_days",
                                     "actual_delivered_days"])

display(basetable)

# COMMAND ----------

display(basetable.select([sum(col(c).isNull().cast("int")).alias(c) for c in basetable.columns]))

print(basetable.count())

# COMMAND ----------

#Removing the remaining NULL Rows
basetable = basetable.dropna(how="any")

# COMMAND ----------

display(basetable.select([sum(col(c).isNull().cast("int")).alias(c) for c in basetable.columns]))

print(basetable.count())

# COMMAND ----------

#Convert categorical variables order_status, DeliveryTimeStatus, order_time_of_day, order_purchase_weekend_or_weekday, delivery_time_of_day and
#order_delivered_weekend_or_weekday into numeric where:
# order_status values 
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline

order_status_IndIndxr = StringIndexer().setInputCol("order_status").setOutputCol("order_status_Ind")

DeliveryTimeStatus_IndIndxr = StringIndexer().setInputCol("DeliveryTimeStatus").setOutputCol("DeliveryTimeStatus_Ind")

order_time_of_day_IndIndxr = StringIndexer().setInputCol("order_time_of_day").setOutputCol("order_time_of_day_Ind")

order_purchase_weekend_or_weekday_IndIndxr = StringIndexer().setInputCol("order_purchase_weekend_or_weekday").setOutputCol("order_purchase_weekend_or_weekday_Ind")

delivery_time_of_day_IndIndxr = StringIndexer().setInputCol("delivery_time_of_day").setOutputCol("delivery_time_of_day_Ind")

order_delivered_weekend_or_weekday_IndIndxr = StringIndexer().setInputCol("order_delivered_weekend_or_weekday").setOutputCol("order_delivered_weekend_or_weekday_Ind")

#------------------------------------

#One-hot encoding of time_of_day and weekend_or_weekday
ohee_catv = OneHotEncoder(inputCols=["order_status_Ind","DeliveryTimeStatus_Ind", "order_time_of_day_Ind", 
                                     "order_purchase_weekend_or_weekday_Ind", "delivery_time_of_day_Ind", "order_delivered_weekend_or_weekday_Ind"],
                          outputCols=["order_status_dum","DeliveryTimeStatus_dum", "order_time_of_day_dum", 
                                      "order_purchase_weekend_or_weekday_dum", "delivery_time_of_day_dum", "order_delivered_weekend_or_weekday_dum"])

pipe_catv = Pipeline(stages=[order_status_IndIndxr, DeliveryTimeStatus_IndIndxr, order_time_of_day_IndIndxr, 
                             order_purchase_weekend_or_weekday_IndIndxr, delivery_time_of_day_IndIndxr, order_delivered_weekend_or_weekday_IndIndxr,
                             ohee_catv])

basetable_v2 = pipe_catv.fit(basetable).transform(basetable)

basetable_v2 = basetable_v2.drop("order_status_Ind","order_status","DeliveryTimeStatus", "order_time_of_day",
                                 "order_purchase_weekend_or_weekday", "delivery_time_of_day","order_delivered_weekend_or_weekday",
                                 "DeliveryTimeStatus_Ind", "order_time_of_day_Ind",
                                 "order_purchase_weekend_or_weekday_Ind", "delivery_time_of_day_Ind", "order_delivered_weekend_or_weekday_Ind",
                                 "order_purchase_time", "order_approved_at_time", "order_delivered_carrier_date_time", "order_delivered_customer_date_time", "order_estimated_delivery_date_time")
display(basetable_v2)

# COMMAND ----------

#Create categorical variables for payment_type and max_payment_sequential and max_payment_installments
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline

payment_type_IndIndxr = StringIndexer().setInputCol("payment_type").setOutputCol("payment_type_Ind")

payment_sequential_IndIndxr = StringIndexer().setInputCol("max_payment_sequential").setOutputCol("payment_sequential_Ind")

payment_installments_IndIndxr = StringIndexer().setInputCol("max_payment_installments").setOutputCol("payment_installments_Ind")

#-------------------------------

#One-hot encoding
ohee_catv = OneHotEncoder(inputCols=["payment_type_Ind", "payment_sequential_Ind", "payment_installments_Ind"],
                          outputCols=["payment_type_dum", "payment_sequential_dum", "payment_installments_dum"])

pipe_catv = Pipeline(stages=[payment_type_IndIndxr, payment_sequential_IndIndxr, payment_installments_IndIndxr, ohee_catv])

basetable_v2 = pipe_catv.fit(basetable_v2).transform(basetable_v2)

basetable_v2 = basetable_v2.drop("payment_type", "max_payment_sequential", "max_payment_installments","payment_type_Ind","payment_sequential_Ind","payment_installments_Ind")

display(basetable_v2)

# COMMAND ----------

# DBTITLE 1,MODEL BUILDING
# Selecting the desired features.
#required_columns = [
 # "order_id", "estimated_delivery_days", "order_to_approval_days", "actual_delivered_days",
 # "order_status_dum","DeliveryTimeStatus_dum", "order_time_of_day_dum",
 # "order_purchase_weekend_or_weekday_dum", "delivery_time_of_day_dum",
 # "order_delivered_weekend_or_weekday_dum",
 # "total_payment_value", "payment_type_dum", "payment_sequential_dum",
 # "payment_installments_dum", "total_items", "total_price", "total_shipping_cost",
 # "average_price", "average_shipping_cost", "average_product_name_lenght",
 # "average_product_description_lenght", "average_product_photos_qty",
 # "average_product_weight_g", "total_amount", "label"
#]

required_columns = [
'order_id',
'estimated_delivery_days',
'order_to_approval_days',
'actual_delivered_days',
'order_status_dum',
'DeliveryTimeStatus_dum',
'order_time_of_day_dum',
'order_purchase_weekend_or_weekday_dum',
'delivery_time_of_day_dum',
'order_delivered_weekend_or_weekday_dum',
'payment_type_dum',
'payment_sequential_dum',
'payment_installments_dum',
'total_payment_value',
'total_items',
'total_price',
'total_shipping_cost',
'average_price',
'average_shipping_cost',
'average_product_name_lenght',
'average_product_description_lenght',
'average_product_photos_qty',
'average_product_weight_g',
'total_amount',
'total_amount_agro_industry_and_commerce',
'total_amount_air_conditioning',
'total_amount_art',
'total_amount_arts_and_craftmanship',
'total_amount_audio',
'total_amount_auto',
'total_amount_baby',
'total_amount_bed_bath_table',
'total_amount_books_general_interest',
'total_amount_books_imported',
'total_amount_books_technical',
'total_amount_cds_dvds_musicals',
'total_amount_christmas_supplies',
'total_amount_computers',
'total_amount_computers_accessories',
'total_amount_consoles_games',
'total_amount_construction_tools_construction',
'total_amount_construction_tools_lights',
'total_amount_construction_tools_safety',
'total_amount_cool_stuff',
'total_amount_costruction_tools_garden',
'total_amount_costruction_tools_tools',
'total_amount_diapers_and_hygiene',
'total_amount_drinks',
'total_amount_dvds_blu_ray',
'total_amount_electronics',
'total_amount_fashio_female_clothing',
'total_amount_fashion_bags_accessories',
'total_amount_fashion_childrens_clothes',
'total_amount_fashion_male_clothing',
'total_amount_fashion_shoes',
'total_amount_fashion_sport',
'total_amount_fashion_underwear_beach',
'total_amount_fixed_telephony',
'total_amount_flowers',
'total_amount_food',
'total_amount_food_drink',
'total_amount_furniture',
'total_amount_furniture_bedroom',
'total_amount_furniture_decor',
'total_amount_furniture_living_room',
'total_amount_furniture_mattress_and_upholstery',
'total_amount_garden_tools',
'total_amount_health_beauty',
'total_amount_home_appliances',
'total_amount_home_appliances_2',
'total_amount_home_comfort_2',
'total_amount_home_confort',
'total_amount_home_construction',
'total_amount_housewares',
'total_amount_industry_commerce_and_business',
'total_amount_kitchen',
'total_amount_luggage_accessories',
'total_amount_market_place',
'total_amount_music',
'total_amount_musical_instruments',
'total_amount_office_furniture',
'total_amount_party_supplies',
'total_amount_perfumery',
'total_amount_pet_shop',
'total_amount_print',
'total_amount_security_and_services',
'total_amount_signaling_and_security',
'total_amount_small_appliances',
'total_amount_small_appliances_home_oven_and_coffee',
'total_amount_sports_leisure',
'total_amount_stationery',
'total_amount_telephony',
'total_amount_toys',
'total_amount_watches_gifts',
'label'
]

# Select only the columns in the features list
model_df = basetable_v2.select(*[col(feature) for feature in required_columns])

display(model_df)

# COMMAND ----------

print(model_df.count())

# COMMAND ----------

model_df.printSchema()

# COMMAND ----------

#Create a train and test set with a 70% train, 30% test split
basetable_train, basetable_test = model_df.randomSplit([0.7, 0.3],seed=123)

# COMMAND ----------

print(basetable_train.count())
print(basetable_test.count())

# COMMAND ----------

from pyspark.ml.feature import RFormula

train = RFormula(formula="label ~ . - order_id").fit(basetable_train).transform(basetable_train)
test = RFormula(formula="label ~ . - order_id").fit(basetable_test).transform(basetable_test)

print("train nobs: " + str(train.count()))
print("test nobs: " + str(test.count()))

# COMMAND ----------

# DBTITLE 1,ALGORITHM - DECISION TREES
# Decision Tree Classifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

dt_model = DecisionTreeClassifier().fit(train)

dt_pred = dt_model.transform(test)

# COMMAND ----------

# DBTITLE 1,Performance Metric - Decision Trees - AUC/ROC, Accuracy, F1 Score
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Initialize a BinaryClassificationEvaluator
binary_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

# Compute the AUC-ROC score
auc_roc = binary_evaluator.evaluate(dt_pred)
print("AUC-ROC:", auc_roc)

# Initialize a MulticlassClassificationEvaluator
multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

# Compute the accuracy score
accuracy = multi_evaluator.evaluate(dt_pred)
print("Accuracy:", accuracy)

print("F1 Score:", MulticlassClassificationEvaluator(metricName="f1").evaluate(dt_pred))

# COMMAND ----------

#Feature Importance 

# Extracing a list of top features from the already trained Decision Tree Model.
import pandas as pd
def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))
  
ExtractFeatureImp(dt_model.featureImportances, train, "features").head(30)

# COMMAND ----------

# Re-running the model again with the top 10 features as mentioned in the step above.

#Based on the above feature selection method, we are only using the top 10 variables for modelling.

DT_features =  ["order_id",
                "estimated_delivery_days",
                "total_items",
                "actual_delivered_days",
                "order_to_approval_days",
                "total_shipping_cost",
                "total_amount_fashion_male_clothing",
                "average_product_name_lenght",
                "total_amount_fixed_telephony",
                "payment_type_dum",
                "payment_installments_dum",
                "label"]


# Select only the columns in the features list
Features_for_DT = basetable_v2.select(*[col(feature) for feature in DT_features])
display(Features_for_DT)

#Create a train and test set with a 70% train, 30% test split
basetable_train_DT, basetable_test_DT = Features_for_DT.randomSplit([0.7, 0.3],seed=123)

print(Features_for_DT.count(),basetable_train_DT.count(),basetable_test_DT.count())

# COMMAND ----------

from pyspark.ml.feature import RFormula

train_DT = RFormula(formula="label ~ . - order_id").fit(basetable_train_DT).transform(basetable_train_DT)
test_DT = RFormula(formula="label ~ . - order_id").fit(basetable_test_DT).transform(basetable_test_DT)

print("train nobs: " + str(train_DT.count()))
print("test nobs: " + str(test_DT.count()))

# COMMAND ----------

# Decision Tree Classifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

dt_model_DT = DecisionTreeClassifier().fit(train_DT)

dt_pred_DT = dt_model_DT.transform(test_DT)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# Initialize a BinaryClassificationEvaluator
binary_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

# Compute the AUC-ROC score
auc_roc = binary_evaluator.evaluate(dt_pred_DT)
print("AUC-ROC:", auc_roc)

# Initialize a MulticlassClassificationEvaluator
multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

# Compute the accuracy score
accuracy = multi_evaluator.evaluate(dt_pred_DT)
print("Accuracy:", accuracy)

print("F1 Score:", MulticlassClassificationEvaluator(metricName="f1").evaluate(dt_pred_DT))

# COMMAND ----------

#create a Cross-Validation model for a Classification model
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

#Perform 5-fold cross-validation
dtc = DecisionTreeClassifier(seed=121)

ccv = CrossValidator(
  estimator = dtc,
  evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC"),
  estimatorParamMaps = ParamGridBuilder().addGrid(dtc.maxDepth, [2, 5]).build(),
  numFolds=2
)

#Fitting the model
cmodel = ccv.fit(train_DT)

# COMMAND ----------

# evaluate the CV model
# Predict
cpred = cmodel.transform(test_DT)

# Evaluate
from pyspark.ml.evaluation import BinaryClassificationEvaluator
BinaryClassificationEvaluator().evaluate(cpred)

# COMMAND ----------

# DBTITLE 1,ALGORITHM - LOGISTIC REGRESSION 
train_lr = RFormula(formula="label ~ . - order_id").fit(basetable_train).transform(basetable_train)
test_lr = RFormula(formula="label ~ . - order_id").fit(basetable_test).transform(basetable_test)

print("train nobs: " + str(train.count()))
print("test nobs: " + str(test.count()))

# COMMAND ----------

#Estimate a logistic regression model
from pyspark.ml.classification import LogisticRegression
logreg_model = LogisticRegression().fit(train)

# COMMAND ----------

#Predict on the test set
logreg_pred = logreg_model.transform(test)
display(logreg_pred)

# COMMAND ----------

display(logreg_pred)

# COMMAND ----------

#Print coefficients
print([logreg_model.intercept,logreg_model.coefficientMatrix.toArray()])
#order of dummies same as order of vectors in features

# COMMAND ----------

#Check the speed at which the model achieves its final result (= plot the values for the loss function)
logreg_model.summary.objectiveHistory
#Reveals whether there are sufficient iterations or whether we need to tune some parameters.

# COMMAND ----------

#Get information on the training of the model
print(logreg_model.summary.accuracy,logreg_model.summary.areaUnderROC)

logreg_model.summary.roc.show()
logreg_model.summary.pr.show()

# COMMAND ----------

# DBTITLE 1,AUC- Logistic Regression Performance Metric - 1
# Evaluate the performance of the model
# BinaryClassificationEvaluator uses the area under the receiver operating characteristic (ROC) curve as the evaluation metric

from pyspark.ml.evaluation import BinaryClassificationEvaluator

BinaryClassificationEvaluator().evaluate(logreg_pred) #AUC is the default

# COMMAND ----------

# DBTITLE 1,Recall - Logistic Regression Performance Metric - 2
# Here me use AreaUnderPR as the metric name.

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Create a BinaryClassificationEvaluator object and set the metric to 'weightedRecall'
evaluator = BinaryClassificationEvaluator(metricName='areaUnderPR')

# Apply the evaluate method to the predictions DataFrame and store the result in a variable
recall = evaluator.evaluate(logreg_pred)

# Print the recall
print("Recall:", recall)

# COMMAND ----------

#Create an output table with order_id, prediction
out_pred = logreg_pred.select("order_id","prediction")

display(out_pred)

# COMMAND ----------

# DBTITLE 1,TEST HANDOUT DATASET
#There are four training data sources:
"""
  products: information about the products
  orders: information about the orders 
  order_items : information about the items ordered
  order_payments : information about the payments
"""
#Load functions
from pyspark.sql.functions import *


# COMMAND ----------

# DBTITLE 1,TEST_ORDER_ITEMS TABLE
"""
Metadata description: 
  •	order_id: Order unique identifier
  •	order_item_id: Sequential number identifying the order of the ordered items. A customer can order multiple items per order
  •	product_id: Product unique identifier
  •	price: user Item price in euro (excl. VAT)
  •	shipping_cost: Cost for shipping the item to the customer in euro (excl. VAT)
 
"""

#Read table: orders_item
order_items_hod=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load("/FileStore/tables/test_order_items.csv")

#print orders_item
display(order_items_hod)

#print metadata of orders_item to check the type of the column
order_items_hod.printSchema()

print(order_items_hod.count())

# COMMAND ----------

#Remove duplicates
order_items = order_items_hod.dropDuplicates()

#Inspect the table
display(order_items_hod.describe())

# COMMAND ----------

# DBTITLE 1,TEST_PRODUCTS
"""
Metadata description: 
product_id: Product unique identifier
product_name_length: Number of characters in the product name
product_description_length: Number of characters in the product description
product_photos_qty: Number of photos included in the product description
product_weight_g: Product weight (in grams)
product_length_cm: Product dimensions - length (in centimeters)
product_height_cm: Product dimensions - height (in centimeters)
product_width_cm: Product dimensions - width (in centimeters)
product_category_name: Product category name
  
"""

#Read table: product
products_hod = spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.option("escape","\"")\
.load("/FileStore/tables/test_products.csv")

display(products_hod)
print("count",products_hod.count())

#products metadata
products_hod.printSchema()

# COMMAND ----------

#inspect the distinct number of rows.
products_hod.distinct().count()

#Meaning the table has unique products id

# COMMAND ----------

# DBTITLE 1,JOINING Order_items_hod and products_hod
order_items_products_hod = order_items_hod.join(products_hod, on="product_id", how="left")
display(order_items_products_hod)

print("count",order_items_products_hod.count())

display(order_items_products_hod.describe())

# COMMAND ----------

#Checking for the NULL Columns

# Count the number of null values in each column

display(order_items_products_hod.select([sum(col(c).isNull().cast("int")).alias(c) for c in order_items_products_hod.columns]))

#inspect the NULL values 

display(order_items_products_hod.filter(col("product_name_lenght").isNull()))


#---------------------------

# replacing NULL values in "product_name_length", "product_description_lenght", "product_photos_qty","product_weight_g","product_length_cm","product_height_cm"
# "product_width_cm","product_category_name"

# compute the mean of each column
col_mean = {col: order_items_products_hod.filter(order_items_products_hod[col].isNotNull()).groupBy(order_items_products_hod[col]).count().agg(mean(order_items_products_hod[col])).first()[0] for col in ["product_name_lenght","product_description_lenght","product_photos_qty","product_weight_g","product_length_cm","product_height_cm","product_width_cm"]}

# # replace NULL values with the mean
order_items_products_hod = order_items_products_hod.na.fill(col_mean)

#Checking the NULL Values once again
# Count the number of null values in each column
display(order_items_products_hod.select([sum(col(c).isNull().cast("int")).alias(c) for c in order_items_products_hod.columns]))




# COMMAND ----------

display(order_items_products_hod.select("product_category_name").distinct())

# Replacing NULL values of "product_category_name" with "Others"

m = order_items_products_hod.groupBy(order_items_products_hod.product_category_name).agg(count(order_items_products_hod.product_category_name).alias("count")).orderBy(desc("count")).first()[0]

print(m)

order_items_products_hod = order_items_products_hod.na.fill(value=m,subset=["product_category_name"])


# Check for NULL values once again.
#Checking the NULL Values once again
# Count the number of null values in each column
display(order_items_products_hod.select([sum(col(c).isNull().cast("int")).alias(c) for c in order_items_products_hod.columns]))

# COMMAND ----------

# removing the duplicates
order_items_products_hod = order_items_products_hod.dropDuplicates()

#check the count
order_items_products_hod.count()

#to check if there is any erroneous values for column price and shipping_cost
from pyspark.sql.functions import *
od= order_items_products_hod.where((col("price")<=0) & (col("shipping_cost")<= 0)).select("order_id", "price", "shipping_cost")
display(od)

print("unique order_ids",order_items_products_hod.select("order_id").distinct().count())

# COMMAND ----------

#count the total number of items bought by the customer per order id
total_items_hod= order_items_products_hod.groupBy("order_id").agg(count("order_item_id").alias("total_items"))
display(total_items_hod)

total_items_hod.count()

# COMMAND ----------

#total price, total shipping cost, average price and average shipping cost of items paid by the customer per order id
total_cost_hod= order_items_products_hod.groupBy("order_id").\
agg(round(sum("price"), 2).alias("total_price"), \
    round(sum("shipping_cost"), 2).alias("total_shipping_cost"), \
    round(avg("price"), 2).alias("average_price"), \
    round(avg("shipping_cost"), 2).alias("average_shipping_cost"),\
    round(avg("product_name_lenght"), 2).alias("average_product_name_lenght"),\
    round(avg("product_description_lenght"), 2).alias("average_product_description_lenght"), \
    round(avg("product_photos_qty"), 2).alias("average_product_photos_qty"),\
    round(avg("product_weight_g"), 2).alias("average_product_weight_g"))
display(total_cost_hod)

total_cost_hod.count()

# COMMAND ----------

#Calculating the total_amount = total_price + total_shipping_cost
total_cost_hod = total_cost_hod.withColumn("total_amount", expr("total_price + total_shipping_cost"))

# COMMAND ----------

dummy_product_category_type_hod = order_items_products_hod.groupBy("order_id").pivot("product_category_name").agg(sum(when(col("price").isNotNull(), col("price")) + when(col("shipping_cost").isNotNull(), col("shipping_cost"))).alias("total_cost")).na.fill(0)

# rename dummy variables column_list = dummy_payment_sequential.columns
column_lists = dummy_product_category_type_hod.columns
prefixs = "total_amount_"
new_column_lists = [prefixs + s for s in column_lists]
new_column_lists


dummy_product_category_type_hod = dummy_product_category_type_hod.toDF(*new_column_lists)


#renaming the Order_id column back 
dummy_product_category_type_hod = dummy_product_category_type_hod.withColumnRenamed("total_amount_order_id","order_id")
display(dummy_product_category_type_hod)

# COMMAND ----------

#merging all the aggregated dataframes together
# which are - total_items + total_cost + dummy_product_category_type 
order_items_products_final_hod = total_items_hod.join(total_cost_hod, on="order_id", how="left")
display(order_items_products_final_hod)

#merging the remaining table
order_items_products_final_hod = order_items_products_final_hod.join(dummy_product_category_type_hod, on="order_id", how="left")
display(order_items_products_final_hod)

#verify the order_id count
print("count",order_items_products_final_hod.select("order_id").distinct().count())

#total rows count
print("distinct count",order_items_products_final_hod.count())

# COMMAND ----------

# DBTITLE 1,TEST_ORDERS TABLE
#Read table: orders
orders_hod = spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.option("escape","\"")\
.load("/FileStore/tables/test_orders.csv")

display(orders_hod)
orders_hod.count()



#Inspect all column datatypes
orders_hod.printSchema()

# COMMAND ----------

# DBTITLE 1,Filtering based on timeline
# selecting orders place between September 2020 and June 2022.
orders_hod= orders_hod.where((col("order_purchase_timestamp")>="2022-07-01 00:00:00") & (col("order_purchase_timestamp")<= "2022-09-30 23:59:59")).select("*")
display(orders_hod)
orders_hod.count()


#count the uniqhe number of order_id
print("count",orders_hod.distinct().count())

#inspect the table
display(orders_hod.describe())

# COMMAND ----------

# Count the number of null values in each column
display(orders_hod.select([sum(col(c).isNull().cast("int")).alias(c) for c in orders_hod.columns]))

orders_hod = orders_hod.filter(when(col("order_id") == "NA", None).otherwise(col("order_id")).isNotNull())

#check the row count after the removal of rows where order_id = "NA"
print("count",orders_hod.count())

# COMMAND ----------

#Convert all date columns from "String" to "Date" format.
from pyspark.sql.types import *

orders_hod= orders_hod.select("*",
col("order_purchase_timestamp").cast("timestamp").alias("order_purchase_time"),
col("order_approved_at").cast("timestamp").alias("order_approved_at_time"),
col("order_delivered_carrier_date").cast("timestamp").alias("order_delivered_carrier_date_time"),
col("order_delivered_customer_date").cast("timestamp").alias("order_delivered_customer_date_time"),                      
col("order_estimated_delivery_date").cast("timestamp").alias("order_estimated_delivery_date_time"))

#As now the date columns type are converted from string to timestamp
orders_hod.printSchema()

# COMMAND ----------

#check for NULL again
display(orders_hod.select([sum(col(c).isNull().cast("int")).alias(c) for c in orders_hod.columns]))



#drop the string date columns
orders_hod= orders_hod.drop("order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date")
orders_hod.printSchema()



#distinct order id
orders_hod.distinct().count()

# COMMAND ----------

# Group the DataFrame by "order_id" and count the number of rows in each group
grouped_df_hod = orders_hod.groupBy("order_id").agg(count("*").alias("count"))

# Filter the groups where the count is greater than 1
filtered_df_hod = grouped_df_hod.filter(grouped_df_hod["count"] > 1)

# Show the filtered DataFrame
filtered_df_hod.show()



#From the above code, we could identify every observation is on "order_id"

# COMMAND ----------

#Find the number of days taken to submit the review after the review was generated for an orderid.
orders_hod =  orders_hod.withColumn("estimated_delivery_days",datediff("order_estimated_delivery_date_time","order_delivered_customer_date_time"))
orders_hod =  orders_hod.withColumn("order_to_approval_days",datediff("order_approved_at_time","order_purchase_time"))
orders_hod =  orders_hod.withColumn("actual_delivered_days",datediff("order_delivered_customer_date_time","order_purchase_time"))


#-------------------

#Creating a new feature(variable) for the difference between estimated delivery and delivery date
#"Early delivery" meant a POSITIVE value
#"On time delivery" meant a ZERO value
#"Late delivery" meant NEGATIVE value

#Creating dummy variable for whether order was "On Time", "Delayed" or "Early"
orders_hod = orders_hod.withColumn("DeliveryTimeStatus", when(col("estimated_delivery_days") > 0, "early")
                              .when(col("estimated_delivery_days") < 0, "delayed")
                              .otherwise("timely"))

#-------------------

#Create a new column "order_time_of_day" that represents the hour of the day at which the customer placed the order. 
# "order_purchase_time" between 06:00:00 and 11:59:59 as "morning"
# "order_purchase_time" between 12:00:00 and 17:59:59 as "afternoon"
# "order_purchase_time" between 18:00:00 and 19:59:59 as "evening"
# "order_purchase_time" between 20:00:00 and 5:59:59 as "night"

from pyspark.sql.functions import when, hour

orders_hod = orders_hod.withColumn("order_time_of_day", 
                        when(hour(orders_hod["order_purchase_time"]).between(6, 11), "morning")
                        .when(hour(orders_hod["order_purchase_time"]).between(12, 17), "afternoon")
                        .when(hour(orders_hod["order_purchase_time"]).between(18, 19), "evening")
                        .otherwise("night"))

#-------------------

#Create a new column "weekend_or_weekday" that represents the day of the week during which the customer purchased the order. 
# "review_answer_time" < 5 as "weekday"
# "review_answer_time" > 5 as "weekend"

from pyspark.sql.functions import when, date_format, dayofweek

orders_hod = orders_hod.withColumn("order_purchase_weekend_or_weekday", 
                        when(dayofweek(orders_hod["order_purchase_time"]) <= 5, "weekday")
                        .otherwise("weekend"))

#-------------------

#Create a new column "delivery_time_of_day" that represents the hour of the day at which the customer recieved the order. 
# "order_purchase_time" between 06:00:00 and 11:59:59 as "morning"
# "order_purchase_time" between 12:00:00 and 17:59:59 as "afternoon"
# "order_purchase_time" between 18:00:00 and 19:59:59 as "evening"
# "order_purchase_time" between 20:00:00 and 5:59:59 as "night"

from pyspark.sql.functions import when, hour

orders_hod = orders_hod.withColumn("delivery_time_of_day", 
                        when(hour(orders_hod["order_delivered_customer_date_time"]).between(6, 11), "morning")
                        .when(hour(orders_hod["order_delivered_customer_date_time"]).between(12, 17), "afternoon")
                        .when(hour(orders_hod["order_delivered_customer_date_time"]).between(18, 19), "evening")
                        .otherwise("night"))


#-------------------

#Create a new column "weekend_or_weekday" that represents the day of the week on which the order was actually delivered. 
# "review_answer_time" < 5 as "weekday"
# "review_answer_time" > 5 as "weekend"

from pyspark.sql.functions import when, date_format, dayofweek

orders_hod = orders_hod.withColumn("order_delivered_weekend_or_weekday", 
                        when(dayofweek(orders_hod["order_delivered_customer_date_time"]) <= 5, "weekday")
                        .otherwise("weekend"))

display(orders_hod)

# COMMAND ----------

int_columns_hod = ["estimated_delivery_days", "order_to_approval_days", "actual_delivered_days"]

for col_name in int_columns_hod:
    orders_hod = orders_hod.withColumn(col_name, col(col_name).cast("double"))

orders_hod.printSchema()

# COMMAND ----------

# DBTITLE 1,TEST_ORDER_PAYMENTS TABLE
"""
Metadata description: 
  •	order_id: Order unique identifier
  •	payment_sequential: Sequential number identifying the order of the payment types used. A customer may use several payment types for one order.
  •	payment_type: Method of payment chosen by the customer
  •	payment_installments: Number of installments chosen by the customer
  •	payment_value: Order value in euro
"""
#Read table: order_payment
test_order_payment=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load("/FileStore/tables/test_order_payments.csv")

#print test_order_payment
display(test_order_payment)

#print schema of test_order_payment to check the type of the column
test_order_payment.printSchema()



#total number of observation(rows) in test_order_payment
print("count", test_order_payment.count())


# COMMAND ----------

#Remove duplicates
test_order_payment = test_order_payment.dropDuplicates()

#Inspect the table
display(test_order_payment.describe())

#unit of analysis is order_id so check how many unique orders there are
test_order_payment.select("order_id").distinct().count()

# COMMAND ----------

#Unique values of column payment type
display(test_order_payment.select("payment_type").distinct().collect())

#Unique values of column payment type
display(test_order_payment.select("payment_installments").distinct().collect())

#Unique values of column payment_sequential
display(test_order_payment.select("payment_sequential").distinct().collect())

# COMMAND ----------

#minimum and maximum payment_sequential and payment_installments choosen by the customer per order id
test_order_payment1= test_order_payment.groupBy("order_id").agg(min("payment_sequential").alias("min_payment_sequential"),                   max("payment_sequential").alias("max_payment_sequential"), min("payment_installments").alias("min_payment_installments"),                       max("payment_installments").alias("max_payment_installments"))
display(test_order_payment1)

#observation
# Filter the rows where col1 and col2 have the same values
test_same_value_rows4 = test_order_payment1.filter(test_order_payment1["min_payment_sequential"] == test_order_payment1["max_payment_sequential"])

# Count the number of rows
test_num_rows4 = test_same_value_rows4.count()

# Print the result
print("Number of rows where col1 and col2 have the same values:", test_num_rows4)

# Filter the rows where col1 and col2 have the same values
test_same_value_rows5 = test_order_payment1.filter(test_order_payment1["min_payment_installments"] == test_order_payment1["max_payment_installments"])

# Count the number of rows
test_num_rows5 = test_same_value_rows5.count()

# Print the result
print("Number of rows where col1 and col2 have the same values:", test_num_rows5)

# we see that there are 12824 unique payments per order_id and almost every order_id has same minimum and maximum sequential and payment
#so we are keeping only the min_payment_sequential and min_payment_installments
test_order_payment1= test_order_payment1.drop("min_payment_sequential", "min_payment_installments")
display(test_order_payment1)

# COMMAND ----------

#total, average, min, max payment value per order id
test_order_payment2= test_order_payment.groupBy("order_id").agg(round(sum("payment_value"), 2).alias("total_payment_value"), round(avg("payment_value"),                           2).alias("average_payment_value"), min("payment_value").alias("min_payment_value"), max("payment_value").alias("max_payment_value"))
display(test_order_payment2)

test_order_payment2.count()

#observations
# Filter the rows where col1 and col2 have the same values
test_same_value_rows6 = test_order_payment2.filter(test_order_payment2["total_payment_value"] == test_order_payment2["average_payment_value"])

# Count the number of rows
test_num_rows6 = test_same_value_rows6.count()

# Print the result
print("Number of rows where col1 and col2 have the same values:", test_num_rows6)

# Filter the rows where col1 and col2 have the same values
test_same_value_rows7 = test_order_payment2.filter(test_order_payment2["total_payment_value"] == test_order_payment2["min_payment_value"])

# Count the number of rows
test_num_rows7 = test_same_value_rows7.count()

# Print the result
print("Number of rows where col1 and col2 have the same values:", test_num_rows7)

# Filter the rows where col1 and col2 have the same values
test_same_value_rows8 = test_order_payment2.filter(test_order_payment2["total_payment_value"] == test_order_payment2["max_payment_value"])

# Count the number of rows
test_num_rows8 = test_same_value_rows8.count()

# Print the result
print("Number of rows where col1 and col2 have the same values:", test_num_rows8)

#since the unique obs. are 12824 per order_id and payment_value that is total, average, min and max are almost same 
test_order_payment2= test_order_payment2.drop("average_payment_value", "min_payment_value", "max_payment_value")

# COMMAND ----------

#join the test_order_payment1 and test_order_payment2 dataframe to create test_order_payment_agg
test_order_payment_agg= test_order_payment1.join(test_order_payment2, "order_id")
display(test_order_payment_agg)

test_order_payment_agg.count()


test_order_payment_agg.printSchema()

# COMMAND ----------

#Creating another intermediate dataframe for fetching payment_type corresponsing to each order_id.
#Payment sequential is unique for each order_id
test_order_payment_group = test_order_payment.select(["order_id","payment_type"])
test_order_payment_group.show(3)
test_order_payment_group = test_order_payment_group.groupby(col("order_id")).agg(first("payment_type").alias("payment_type"))
test_order_payment_group.show(5)
print(test_order_payment_group.count())

# COMMAND ----------

# merging it with the original dataframe 
test_order_payment_final= test_order_payment_group.join(test_order_payment_agg, "order_id", "left")

display(test_order_payment_final)

# COMMAND ----------

# DBTITLE 1,COUNT OF ALL DATATABLES USED ABOVE
#Checking the count of every table
print(
      "order_items_products_final_hod",order_items_products_final_hod.count(),\
      "orders_hod",orders_hod.count(),\
      "test_order_payment_final", test_order_payment_final.count())

# COMMAND ----------

#create basetable by joining tables in the following order:
# 1. order_items_products_final_hod
# 2. orders_HOD
# 3. test_order_payment_final


test_basetable= orders_hod.join(test_order_payment_final, "order_id", "left")

test_basetable= test_basetable.join(order_items_products_final_hod, "order_id", "left")

display(test_basetable)

# COMMAND ----------

#Inspect for NULL values again
# Count the number of null values in each column

display(test_basetable.select([sum(col(c).isNull().cast("int")).alias(c) for c in test_basetable.columns]))

# COMMAND ----------

# treat the null values of the datediff columns with "99999", as null cannot be given to the model 
# and also to identify where we had the null values for these columns

test_basetable_v1 = test_basetable.na.fill(value=99999,subset=[
                                     "estimated_delivery_days",
                                     "order_to_approval_days",
                                     "actual_delivered_days"])

display(test_basetable_v1)

# COMMAND ----------

#Inspect for NULL values again
# Count the number of null values in each column

display(test_basetable_v1.select([sum(col(c).isNull().cast("int")).alias(c) for c in test_basetable_v1.columns]))

# COMMAND ----------

test_basetable_v1.count()
test_basetable_v2 = test_basetable_v1.dropna(how="any")

# COMMAND ----------

#check basetable count after dropping NA values
test_basetable_v2.count()

#Inspect for NULL values again
# Count the number of null values in each column

display(test_basetable_v2.select([sum(col(c).isNull().cast("int")).alias(c) for c in test_basetable_v2.columns]))

# COMMAND ----------

display(test_basetable_v2)

# COMMAND ----------

#Convert categorical variables order_status, DeliveryTimeStatus, order_time_of_day, order_purchase_weekend_or_weekday, delivery_time_of_day and
#order_delivered_weekend_or_weekday into numeric where:
# order_status values 
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline

# order_status_IndIndxr = StringIndexer().setInputCol("order_status").setOutputCol("order_status_Ind")

DeliveryTimeStatus_IndIndxr = StringIndexer().setInputCol("DeliveryTimeStatus").setOutputCol("DeliveryTimeStatus_Ind")

order_time_of_day_IndIndxr = StringIndexer().setInputCol("order_time_of_day").setOutputCol("order_time_of_day_Ind")

order_purchase_weekend_or_weekday_IndIndxr = StringIndexer().setInputCol("order_purchase_weekend_or_weekday").setOutputCol("order_purchase_weekend_or_weekday_Ind")

delivery_time_of_day_IndIndxr = StringIndexer().setInputCol("delivery_time_of_day").setOutputCol("delivery_time_of_day_Ind")

order_delivered_weekend_or_weekday_IndIndxr = StringIndexer().setInputCol("order_delivered_weekend_or_weekday").setOutputCol("order_delivered_weekend_or_weekday_Ind")

#------------------------------------

#One-hot encoding of time_of_day and weekend_or_weekday

ohee_catv = OneHotEncoder(inputCols=[ "DeliveryTimeStatus_Ind", "order_time_of_day_Ind", 
                                     "order_purchase_weekend_or_weekday_Ind", "delivery_time_of_day_Ind", "order_delivered_weekend_or_weekday_Ind"],
                          outputCols=[ "DeliveryTimeStatus_dum", "order_time_of_day_dum",
                                     "order_purchase_weekend_or_weekday_dum", "delivery_time_of_day_dum", "order_delivered_weekend_or_weekday_dum"])

pipe_catv = Pipeline(stages=[ DeliveryTimeStatus_IndIndxr, order_time_of_day_IndIndxr, 
                             order_purchase_weekend_or_weekday_IndIndxr, delivery_time_of_day_IndIndxr, order_delivered_weekend_or_weekday_IndIndxr,
                             ohee_catv])

test_basetable_v3 = pipe_catv.fit(test_basetable_v2).transform(test_basetable_v2)

# test_basetable_v3 = test_basetable_v2.drop("DeliveryTimeStatus", "order_time_of_day",
#                                  "order_purchase_weekend_or_weekday", "delivery_time_of_day","order_delivered_weekend_or_weekday",
#                                  "DeliveryTimeStatus_Ind", "order_time_of_day_Ind",
#                                  "order_purchase_weekend_or_weekday_Ind", "delivery_time_of_day_Ind", "order_delivered_weekend_or_weekday_Ind",
#                                 "order_purchase_time", "order_approved_at_time", "order_delivered_carrier_date_time",   "order_delivered_customer_date_time", "order_estimated_delivery_date_time")

display(test_basetable_v3)

# COMMAND ----------

#Create categorical variables for payment_type and max_payment_sequential and max_payment_installments
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline

payment_type_IndIndxr = StringIndexer().setInputCol("payment_type").setOutputCol("payment_type_Ind")

payment_sequential_IndIndxr = StringIndexer().setInputCol("max_payment_sequential").setOutputCol("payment_sequential_Ind")

payment_installments_IndIndxr = StringIndexer().setInputCol("max_payment_installments").setOutputCol("payment_installments_Ind")
#-------------------------------
#One-hot encoding
ohee_catv = OneHotEncoder(inputCols=["payment_type_Ind", "payment_sequential_Ind", "payment_installments_Ind"],
                          outputCols=["payment_type_dum", "payment_sequential_dum", "payment_installments_dum"])

pipe_catv = Pipeline(stages=[payment_type_IndIndxr, payment_sequential_IndIndxr, payment_installments_IndIndxr, ohee_catv])

test_basetable_v3 = pipe_catv.fit(test_basetable_v3).transform(test_basetable_v3)

# test_basetable_v3 = test_basetable_v3.drop("payment_type", "max_payment_sequential", "max_payment_installments","payment_type_Ind","payment_sequential_Ind","payment_installments_Ind")

display(test_basetable_v3)

# COMMAND ----------

# DBTITLE 1,PREDICTING ON THE TEST_HANDOUT_DATA (Basetable_v2)
#fetching the list of features based on the forward feature selection

#Based on the above feature selection method, we are only using the top 10 variables for modelling.
TestHandOut_features =  ["order_id",
                         "estimated_delivery_days",
                         "total_items",
                         "actual_delivered_days",
                         "order_to_approval_days",
                         "total_shipping_cost",
                         "total_amount_fashion_male_clothing",
                         "average_product_name_lenght",
                         "total_amount_fixed_telephony",
                         "payment_type_dum",
                         "payment_installments_dum"
                        ]


# Select only the columns in the features list
TestHandOutFeaturesDF = test_basetable_v3.select(*[col(feature) for feature in TestHandOut_features])
display(TestHandOutFeaturesDF)

#------------------------------------------------

#Use Vector Assembler to club al the variables together into a single column called "features"
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
  inputCols=[col for col in TestHandOutFeaturesDF.columns if col != "order_id"],
  outputCol="features")

TestHandOutFeaturesDF = assembler.transform(TestHandOutFeaturesDF)

# COMMAND ----------

display(TestHandOutFeaturesDF)

# COMMAND ----------

#Subset to fetch only the "order_id" and "features" columns

TestHandOutFeaturesDF = TestHandOutFeaturesDF.select(["order_id","features"])

display(TestHandOutFeaturesDF)

# COMMAND ----------

#appling MODEL on the HANDOUT DATA SET and storing it in "Handout_predictor"
Handout_predictor = dt_model_DT.transform(TestHandOutFeaturesDF)

# COMMAND ----------

#inspect the columns
Handout_predictor.printSchema()

Handout_predictor.count()

# COMMAND ----------

#subsetting the desired columns
output = Handout_predictor.select("order_id", "prediction")

display(output)

# COMMAND ----------

# DBTITLE 1,Visualizations


# COMMAND ----------

# DBTITLE 1,Review Score as per timely delivery of order
display(basetable)

# COMMAND ----------

# DBTITLE 1,Review Score as per delivery time of order
display(basetable)

# COMMAND ----------

# DBTITLE 1,Review Score as per Weekday or Weekend delivery of the order
display(basetable)

# COMMAND ----------

# DBTITLE 1,Review Score as per Type of Payment
display(basetable)

# COMMAND ----------

# DBTITLE 1,Review Score as per Average Price of Order based on timely delivery of order
display(basetable)
