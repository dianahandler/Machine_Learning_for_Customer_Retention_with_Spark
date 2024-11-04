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

