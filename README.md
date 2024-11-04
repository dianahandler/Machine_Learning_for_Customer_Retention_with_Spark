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
  
Outcome:  
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


## Scaler & One More Vector Assembler


We utilize StandardScaler() to standardize our features by subtracting the mean and dividing by the standard deviation  

Next, we add our customer_type_encoded feature using VectorAssembler() again to our vector of numerical features to give us our complete input features vector

![image](https://github.com/user-attachments/assets/a5e303e9-2974-4ce6-b720-dc5078260ca3)

## Create Pipeline Stages

We build our pipeline using all of our transformations that can be executed all at once using the Pipeline() class. We create the class and use set.Stages() to specify the sequence of stages to be used in our pipeline. Each stage is a transformer of estimator and will be executed in the order specified.  
 
We have our indexer where we convert our customer_type string categorical column to a numerical index, our encoder where we encode it into one hot encoded vector of values, our numerical assembler takes in our numerical features and outputs it into a single-vector column, our scaler standardizes the output features and we utilize assembler(vectorassembler) to add our one hot encoded column to our vector.

![image](https://github.com/user-attachments/assets/7115d9d7-deca-4276-b977-0407c89160f1)


Next, we split our data into training and test data sets. Were gonna fit our pipeline to our TRAINING dataset, then transform BOTH of the trainins and test datasets with the parameters gathered from that fit. We split our dataset so that 80% of the data will be allocated to the training set and 20% to the testing. Seed=2022 is a parameter that sets a random seed for reproducibility. Using a seed ensures that the split will be the same each time which is helpful for consistent results in experiments.
We get back the sizes of the two datasets  
 
Next, we fit the entire pipeline to the training data “train”. Each stage of the pipeline is applied to the training data.  

After fitting, pipeline_model contains the fitted model for all stages which has the info for transforming new data. So we run pipeline_model on both the train and test dataset.  
 
The transformed Datasets are now ready for training a machine learning model as they include all the necessary feature engineering steps performed consistently on both datasets.   


## Create, Train, and Evaluate a Logistic Regression Model


Next, we initialize a LogisticRegression() model and specify the column that contains the feature vector used for predictions, our labelCol which specifies the label or TARGET/dependent variable for the model in our case is repeat_customer, and maxIter which specifies the max number of iterations to formulate our model. So we pick 5 to find the optimal coefficients
We fit the logistic regression model to the transformed dataset(train_transformed). The model learns the relationship between the features and the target variable by estimating coefficients for the logistic function.  
 
Next, we have predictions=lr.model.transform(test_transformed) which applies the TRAINED logistic regression model to the transformed TESTING dataset.
The Transform() method here generates predictions for each instance in the test set based on the features in all_features column. We select the columns we want to analyze/view.  

rawPrediction: this column contains the raw output of the model before applying the softmax function, un-normalized scored for each class
prediction: This column contains the predicted class label(0 or 1) for each instance based on the model’s output
Probability: this column provides the predicted probabilities for each class indicating the model’s confidence in its predictions. The probability of being a repeat customer and the probability of being a non-repeat customer.  


![image](https://github.com/user-attachments/assets/e9c14ec4-d922-497d-a503-6c6b682bc790)


## Test Area Under ROC, Accuracy, Weights



We will use an evaluator BinaryClassificationEvaluator() to see how well our model is performing. We set our parameters where labelCol is our target variable and has the actual labels(ground truth) for the test set, that indicates whether each customer is a repeat customer or not. And we specify the rawPredictionCol that contains our raw prediction scores from logistic regression model.  

The evaluate method of the evaluator instance is called with predictions as the argument. This will calculate the area under the ROC (Receiver Operating Characteristic curve()AUC) which is common metric to evaluate performance of binary classification models.
The AUC ranged from 0 to 1, 1 indicating perfect classification, .5 is indiscriminate(might as well guess), and below .5 is worse-than-random performance.  

We generate an Area Under the ROC Curve(or AUC-ROC) of 0.97 which suggests that the model is very good at correctly classifying the positive class(repeat customers) and the negative class(non-repeat).   
 
Next we filter our predictions dataframe to include only rows where the actual value of repeat_customer matches the predicted value.  
Accuracy = # of Correct Predictions/Total Predictions

We generate an accuracy of .92 which suggests that the model is highly effective in classifying customers as repeat or non repeat.   
This would suggest that we can trust this model to provide reliable predictions.  

IMPORTANT TO CONSIDER OTHER METRICS like AUC, precision, recall to get fuller picture of model performance especially is class distribution is imbalanced.  

Finally, we inspect the model coefficients. This code retrieves the coefficients(or WEIGHTS) of the logistic regression model(lr_model). These weights indicate the importance and impact of each feature on the prediction outcome.
We create a dataframe that converts the weights to float values, set the column name for the dataframe and set the index of the DF to the corresponding feature names.  


![image](https://github.com/user-attachments/assets/f60c6dc3-6041-4c17-a354-b67e72fcbfec)


If feature weight is positive, this suggests that as that feature value increases, so the does the likelihood of the customer being a repeat buyer.  

Inversely, if the weight is negative, it implies that as that feature value decreases, the likelihood of that customer being a repeat buyer decreases as well.  

- Days_until_shipped weight/coefficient is -0.935 which indicates that as the number of days until an order is shipped increases, the likelihood of a customer being a repeat buyer decreases.
	-  So a potential implication is customers prefer quicker shipping times and longer shipping times reduces their likelihood of being a repeat customer

- Total weight/coefficient is 3.122 thus a positive weight indicates that as total amount spent increases, the likelihood of the customer being repeat increases as well. 
	-  Implication:Higher spending may correlate with greater customer satisfaction/loyalty

- Customer_type_index weight/coefficient is -0.750007 suggests that being classified in one category is associated with a lower likelihood of being a repeat buyer
	-  Could imply that non-members have a lower retention rate


## Create and Preprocess Recent Customer Data/Make PRedictions

- This model was run on the recent_combined_table dataframe and similar predictions were generated  
- The distribution of repeat vs non-repeat was compared and displayed a disparity between the two datasets  
- Implication that current model may not be as effective and changes can be made

![image](https://github.com/user-attachments/assets/1d1a6b84-b336-430d-b487-c810ca187593)


![image](https://github.com/user-attachments/assets/13473563-facd-4aa8-8374-a8a5a9c3e68e)

We see here that there is a much greater number of non repeat customer vs repeat in our historical dataset meanwhile when we ran our model on the recent_customers table, it was nearly a 50/50 likelihood of being a repeat customer.  
 
This suggested to me that our model isn’t as reliable as I originally thought so in the future, it might be good to add or modify features to better capture some nuances of repeat purchasing behavior or RESAMPLING techniques like SMOTE which generates synthetic examples of the minority class to create a more balanced dataset.


## Considerations/Methods to Improve

- Data Quality
- Feature Engineering
- Feature Selection
- Class balance

-  Add/modify features to better capture some nuances in the data
-  Resampling techniques(SMOTE)
-  Conducted some additional testing, converting product_line and extracting store_location from invoice_id to create two new numerical indices
-  Two additional features could potentially improve the model


Here we have added product_line and store_location as numerical indices to view their correlations  
Adding new features can allow us to use a more robust dataset for a more effective ML model


![image](https://github.com/user-attachments/assets/a383401f-2666-4ff4-82cd-60c8a6aa1688)







