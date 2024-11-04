# Machine Learning for Customer Retention with Spark
This project employs machine learning to predict customer retention for Distributed Discounts. Using Spark, we analyzed historical customer and transaction data to build a logistic regression model that identifies potential repeat buyers among first-time customers, aiding in targeted marketing strategies.

Overview
This project leverages Spark and machine learning to predict whether first-time customers will become repeat buyers for Distributed Discounts. By analyzing historical customer and transaction data, we aim to inform targeted advertising strategies.

Steps Taken
Data Ingestion:

Created a SparkSession and imported data from capstone_customers.csv and capstone_invoices.csv into Spark DataFrames, ensuring the appropriate schema was applied.
Data Preparation:

Cleaned and preprocessed the labeled dataset. This involved handling missing values and encoding categorical variables.
Developed a preprocessing pipeline to manage non-numeric features and eliminated certain features deemed irrelevant for model accuracy.
Data Splitting:

Divided the labeled dataset into training and testing sets to validate model performance.
Model Training:

Trained a logistic regression model on the training data and evaluated its performance using the test set.
Predictions:

Applied the trained model to predict repeat purchase behavior for customers in the capstone_recent_customers.csv dataset.
Outcome
The project successfully generated predictions for first-time customers, enabling Distributed Discounts to identify potential repeat buyers and enhance their marketing strategies.


## Outlined Steps w/ Code:


Create a SparkSession and read in our data as Spark Dataframes
Inspect first 5 rows of each CSV
Count rows of each clean dataframe vs. unclean
Drop null values
Inspect schema of each dataframe

![image](https://github.com/user-attachments/assets/4b78c54e-e765-4822-85cb-0b9242359504)

![image](https://github.com/user-attachments/assets/8a0251ca-bcd9-4b26-a3c0-73e32aa875c1)


## Updating Schemas/Changing Data Types

We inspect the schemas of each table and determine which columns to update datatypes(from string to integer)

![image](https://github.com/user-attachments/assets/94d1c366-a510-4f0b-8b58-c89b6cc01d0f)

For each dataframe, we define a new Schema using Spark SQL and apply the schema to our four datasets
We utilize printSchema() to ensure that our schema has been correctly updated
This was done for all four datasets


![image](https://github.com/user-attachments/assets/43d80326-a614-4d1a-8006-2d3e95df314d)


## Utilize Spark SQL for Queries/Joins

We create a temporary view using createOrReplaceTempView() from our dataframe that will act as a table that we can query for customers w/ SQL-like interface
We repeat this process with the invoices_data and perform an inner join on the customer_id column


![image](https://github.com/user-attachments/assets/d2a784db-77c5-4e76-b697-a097ce0d9d2e)


The resulting combined_table is displayed to confirm our two datasets were successfully joined
This process is repeated with the recent_customers and recent_invoices dataframes as well


![image](https://github.com/user-attachments/assets/720c32c7-2b22-4e9d-b417-288b9a3e4543)

## Create Preprocessing Pipeline to Handle Non-Numeric Features  
- Create StringIndexer
- Encoder
- Numerical Assembler
- Scaler
- Assembler


## MLlib Overview  
In Spark’s MLlib, Logistic Regression is a statistical method used for binary classification problems. It predicts whether an instance belongs to one of two categories(yes or no, repeat or non-repeat custome, etc). It uses the logistic function to model the relationship between the input features(like total purchases and days until shipped) and the probability of the target class(repeat customer)\. The output is a probability between 0 and 1  
  
How it works in MLlib:  
- Input Features: you provide a set of features(NUMERIC)columns that help the model make predictions
- Training: the model learns from your ‘labeled’ data(where the target variable is KNOWN to adjust its parameters.
- Prediction: Once traines, you can use the model to predict the probability that new, unseen instances belong to the positive class(repeat customer)
- Evaluation: You can evaluate the model’s performance using metrics like accuracy, precision, AUC-ROC to see how well it will predict

With respect to OUR data:  
Target Variable/Label = repeat_customer  
Columns to exclude: customer_id, product_line, product_id, invoice_id

## Indexer

We evaluate our categorical data and use StringIndexer() to convert the categorical string values into numerical indices
We convert customer_type from a string to numerical index with non-member as 1 and member as 0  
A new column customer_type_index is created and added to the dataframe
Indexer.fit fits the indexer to the combined_table, which computes the indices for the distinct values in the customer_type column  
We utilize TRANSFORM after fitting to add a new column, customer_type_index which contains the new numeric index.  

![image](https://github.com/user-attachments/assets/37357278-845d-42de-9ae2-91ca5148010b)

## Feature Correlation Matrix


We generate a correlation matrix with our 3 numeric features including customer_type_index  


![image](https://github.com/user-attachments/assets/9d0a0fcc-0d19-4950-a002-d7366944a1b1)


![image](https://github.com/user-attachments/assets/0d4ea58b-5166-4200-aa98-b9f361ef2c46)


Correlation Coefficient Interpretations:  
- 1 indicates that as one increases so does the other
- -1 indicates that as one increases, the other decreases 
- 0 indicates no correlation
  
Feature Correlation Matrices may be insightful as to which features are going to be more influential in predicting our target variable  

## Correlation Matrix WITH Target Variable

The same process was repeated however this time we include the target variable repeat_customer to see the relationships between our features  

![image](https://github.com/user-attachments/assets/aca3784d-7009-44ac-9569-36c912da8402)  

We see customer_type_index has a slightly negative correlation coefficient and it was determined that it would be a good idea to include it as it may just be capturing different insights than days_until_shipped and total and also might balance their high correlation coefficient out a little and generally have a more robust dataset to work with  


## OneHotCodeEstimator & VectorAssembler

We utilize ONEHOTENCODERESTIMATOR() and we create a new column customer_type_encoded that consists of a one hot encoded vector of values.  
This is necessary as we need to put our index in a format that our machine learning algorithm can read.  
Encoder.fit fits the encoder to combined_table_indexed dataframe, determining the on-hot encoding based on the distinct values in customer_type_encoded  
We utilize TRANSFORM to add a new column with our data in customer_type_encoded. Thus we are transforming customer_type_index into a binary vector format  


Vector Assembler:  
Next, we transform our numerical input features into a vector of features that we will just call “features”. We must do this transformation bc MLlib algorithms operate on a single vector of input features as opposed to a list.  
VectorAssmbler() is a feature transformer in Spark MLlib that combines multiple columns into a single vector column.  
Why single vector? MLlib algorithms expect input in the form of a single vector for each data point rather than separate columns or lists bc the algorithms are designed to take in data as a whole, which makes computations more efficient. Additionally, many machine learning algorithms INCLUDING logistic regression use linear algebra operations which work naturally with vectors.


![image](https://github.com/user-attachments/assets/28d17d9a-8a6c-4f91-a5ac-5f880cfda6c1)



























