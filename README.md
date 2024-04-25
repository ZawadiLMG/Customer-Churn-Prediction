# Telcom Customer-Churn-Prediction
## A Machine Learning Model (MLM) for Predicting Customer Churn

Customer churn otherwise known as customer attrition or customer turnover is the percentage of customers that stopped using your company’s product or service within a specified time frame.
For instance, if you began the year with 500 customers but later ended with 480 customers, the customer churn rate would be 4%. This is the percentage of customers that left and stopped using the company’s products.

If we could figure out why a customer leaves and when they leave with reasonable accuracy, it would immensely help the organization to create strategies for retention.

Leveraging machine learning models can provide a solution by predicting potential churners based on various factors, including usage patterns, payment history, and demographic data.

## Project Overview
Telcom Corporation seeks to understand the lifetime value of each customer and know what factors affect the rate at which customers stop using their network or churn. The company would like to build a model that predicts whether a customer will churn or not.

The project uses machine learning models and follows the CRISP-DM methodology as follows:


- Business understanding — What does the business need?
- Data understanding — What data do we have / need? Is it clean?
- Data preparation — How do we organize the data for modeling?
- Modeling — What modeling techniques should we apply?
- Evaluation — Which model best meets the business objectives?
- Deployment — How do stakeholders access the results?

## 1.1 Business Understanding
### Goal: To build a machine learning model that predicts whether a customer will churn or not

### Objective: To find the key indicators of churn as well as the retention strategies that can solve this problem

### Hypothesis

Null Hypothesis: There is no statistically significant relationship between contract type and customer churn

Alternative hypothesis: There is a statistically significant relationship between contract type and customer churn

### Analytical Questions

1. How do monthly charges and total charges impact customer churning?

2. How does tech support influence the likelihood of a customer to churn or not?

3. How does tenure impact customer churning?

4. How does internet service type impact churning?

5. How does the payment method impact churning?

6. How does contract type impact customer churning?

7. Which gender is churning at a higher rate?

8. Does having a partner affect churning?

## 1.2 Data Understanding
There are three sets of data for this project. The first dataset is a CSV file located in an SQL database, while the second is an Excel file. Both datasets are training data for the models. The third dataset, a CSV file, is the test data.

We began by importing the necessary packages for the project. This includes Pandas, Numpy, Matplotlib.pyplot, Seaborn, Joblib, pyodbc, etc.

After loading the datasets, we concatenated the first two datasets into one dataframe, under the name ‘training_data”.

Next, we performed an exploratory data analysis (EDA).

### Exploratory Data Analysis (EDA)
In the EDA, data was explored for nulls, duplicates, column data types, column contents, and overall structure.

The following was noted:

- There are 5043 entries in the training data and 2000 in the test data

- Multiple Lines, OnlineSecurity, OnlineBackup, Deviceprotection, TechSupport, StreamingTV, StreamingMovies, and Churn columns from traiing_data have missing values

- There are no missing values in test data

- TotalCharges column has object data type, but contains numbers

- Column names contain capital letters

Statistical description of the training_data also revealed that it is highly scaled, as seen below:


### Data Cleaning
The following actions were taken to clean the data:

i. Dropping of CustomerID column: This column is unnecessary and won’t be used in the project

ii. Converting TotalCharges column to numeric type from object — It contains numbers (sums of money)

iii. Changing column names to small letters — Due to case sensitivity, it is better to mantain all column names in small letters throughout. I.E. ‘TotalCharges’ to ‘totalcharges’

After cleaning the training_data, we performed univariate, bivariate, and multivariate analyses to understand relationships between variables in the data. The results are as follows:

### Univariate Analysis
Density Distribution: The data is positively skewed.

Outliers
There were no outliers in the numerical columns
There was one outlier in the column ‘seniorcitizen’

### Bivariate Analysis
Total charges is highly correlated with tenure (0.83)
Total charges is also correlated with monthly charges (0.65)
Lowest correlation is between tenure and senior citizen (0.0046)
Customers with two-year contracts are the least likely to churn
Customers with month-month contracts are the most likely to churn
The shorter the contract, the more likely that a customer will stop using the network
Majority of customers have total charges below 2000
Majority of churning customers are clustered below total charges of 1000
This implies that customers with total charges above 2000 are less likely to churn.
Majority of customers who churn stay in the company for a period less than 20 months
Less customers tend to churn if they have stay in the company for more that 40 months
Customers who have stayed in the company for 60–80 months have the lowest likelihood of churning and the highest retention

### Multivariate Analysis
There’s a positive linear relationship between totalcharges and tenure, total charges and monthly charges, and a strong relationship between monthly charges and tenure

Next, we answer a set of analytical questions to further understand the data.

Analytical Questions
1. How do monthly charges and total charges impact customer churning?

- Customers with monthly charges of at least 20 churn at the lowest rate

- Customers with monthly charges between 80 and 100 churn at the highest rate

2. How does tech support influence the likelihood of a customer to churn or not?

- Customers without tech support have the highest and lowest rates of churning.

- For customers with tech support, only 250 churned while more than 1200 did not churn

- For customers with no internet service, the churn rate was very low. Less than 100 customers churned while more than 200 did not.

This implies that the contract type is a significant predictor of churn.

3. How does tenure impact customer churning?

- New customers churn at a higher rate than older customers

- Customers churn more between 0–20 weeks

- Older customers are less likely to churn

- Customers between 40 and 70 weeks churn at the lowest rate

- Customers churn less over time

4. How does internet service type impact churning?

- Customers using fiber optic churn more in comparison to those using DSL and those without. 951 customers with fiber optic churned while only 309 with DSL churned

- DSL internet service has the highest customer retention, retaining 1406 customers and only churning 309. Fiber optic , on the other hand, is the worst performing service, retaining 1296 and churning 951.

- Customers without internet service also had a good retention rate, where only 76 churned while 1004 retained.

5. How does the payment method impact churning?

- Customers who pay thorugh credit card (automatic) have the best retention rate. 922 were retained while 168 churned.

- Customers who pay via by bank transfers (916 retained, 212 churned) and mailed checks (927 retained, 198 churned) have the second best retention rates.

- Customers who pay via electronic check show the poorest retention (941 retained versus 758 churned)

6. How does contract type impact customer churning?

While customers with month-month contracts are the majority, those with two-year contracts have the lowest churn rate.
Month-month contract customers churn at almost a similar rate as those who remain using Telcom’s network
7. Which gender is churning at a higher rate?

- Male and female customers are churning at nearly the same rate

- Among female customers, 1823 remained while 661 churned

- Among male customers, 1883 remained while 675 churned

- Gender might not a significant predictor of churn

8. Does having a partner affect churning?

More customers without partners churned compared to those with customers, but the difference is very small
Partner might not be a significant predictor of churn
Next, we tested our hypothesis.

Hypothesis
Null Hypothesis: There is no statistically significant relationship between contract type and customer churn

Alternative hypothesis: There is a statistically significant relationship between contract type and customer churn

The case here is to test for the association of these two variables.

Chi-Square test was used and the following results obtained:

It was concluded that a significant relationship exists between contract type and churn.

## 1.3 Data Preparation
Data was checked for balance and a moderate imbalance was noted. The ratio of the minority to the majority class was 1336:3707.

It was then split into training and evaluation data (X and y).


Next, we prepared pipelines for the data, where we cleaned it using the SimpleImputer. Both numerical and categorical pipelines were prepared.

With the final pipeline and preprocessor ready, we proceeded to modelling. Note that models will be trained on both balanced and imbalanced datasets.

## 1.4 Modelling
### On Imbalanced Dataset

Four models were trained for this data: Logistic Regression, Random Forest, KNN, and Decision Tree.


#### Key Insights

- Logistic Regression achieved the highest F1 score among all models, indicating good overall performance in terms of precision and recall.

- Random Forest achieved a slightly lower F1 score compared to Logistic Regression but still performed well.

- K-Nearest Neighbors (KNN) achieved the lowest F1 score among all models, indicating slightly lower overall performance compared to Logistic Regression and Random Forest.

- Decision Tree achieved the lowest F1 score among all models, indicating the lowest overall performance in terms of precision and recall

### On Balanced Dataset

The same four models were trained on the balanced dataset.

For the balanced data, Random Forest is the best perfoming model with logistic regression taking a second place, while KNN is the poorest performing model

## Model Evaluation

Evaluation of the models was done using an ROC-AUC curve. Logistic regression is the best performing model, closely followed by Random Forest.

#### Hyperparameter Tuning

After tuning hyperparameters there was a significant increase in the f1 score. Logistic regression emerged as the best model, followed by Random Forest

#### Testing the Model

We tested the logistic regression model  and found that among the 2000 customers who entries are in the test data file, 824 customers will most likely stop using the network, while 1176 will be retained.

### Recommendations

1. Invest in developing and promoting long-term contracts: Longer tenure reduces the likelihood of the predicted outcome, suggesting that longer-term customers are more stable.

2. Invest more in fiber optic services: Customers with fiber optic internet service are more likely to churn.

3. Provide customers with high total charges bonuses and discounts: As total charges increase, the odds of churning increase

### Conclusion

We have created and saved the trained models to make predictions in the future using them. You can view the notebook used in this project on my Github.

Thank you for checking out my project :)

Contact me with any enquiries: zawadilois@gmail.com