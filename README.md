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


# Updating Schemas/Changing Data Types

We inspect the schemas of each table and determine which columns to update datatypes(from string to integer)

![image](https://github.com/user-attachments/assets/94d1c366-a510-4f0b-8b58-c89b6cc01d0f)

For each dataframe, we define a new Schema using Spark SQL and apply the schema to our four datasets
We utilize printSchema() to ensure that our schema has been correctly updated
This was done for all four datasets


![image](https://github.com/user-attachments/assets/43d80326-a614-4d1a-8006-2d3e95df314d)


# Utilize Spark SQL for Queries/Joins

We create a temporary view using createOrReplaceTempView() from our dataframe that will act as a table that we can query for customers w/ SQL-like interface
We repeat this process with the invoices_data and perform an inner join on the customer_id column


![image](https://github.com/user-attachments/assets/d2a784db-77c5-4e76-b697-a097ce0d9d2e)


The resulting combined_table is displayed to confirm our two datasets were successfully joined
This process is repeated with the recent_customers and recent_invoices dataframes as well


![image](https://github.com/user-attachments/assets/720c32c7-2b22-4e9d-b417-288b9a3e4543)

# Create Preprocessing Pipeline to Handle Non-Numeric Features  
- Create StringIndexer
- Encoder
- Numerical Assembler
- Scaler
- Assembler
