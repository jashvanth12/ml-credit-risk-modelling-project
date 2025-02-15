# credit-risk-modeling

# Introduction

Credit risk modelling is a critical aspect of financial risk management, enabling lenders and financial institutions to assess the likelihood of a borrower defaulting on a loan. The primary objective of this project is to develop a predictive model that estimates the probability of credit risk, assigns a credit score, and determines a risk rating for each applicant.

# Objective

The goal of this project is to analyze customer data, loan details, and credit bureau information to predict the creditworthiness of individuals. By leveraging machine learning techniques, we aim to build a model that can accurately predict the probability of default, categorize customers based on their credit scores, and provide a comprehensive risk rating system.

# Dataset Overview

This project utilizes three key datasets:

# Customer Data (df_customers)

Contains demographic and personal details about customers.

Columns:

* cust_id (Customer ID)

* age

* gender

* marital_status

* employment_status

* income

* number_of_dependants

* residence_type

* years_at_current_address

* city

* state

* zipcode

# Loan Data (df_loans)

Provides details about the loans sanctioned to customers.

Columns:

* loan_id (Loan ID)

* cust_id (Customer ID)

* loan_purpose

* loan_type

* sanction_amount

* loan_amount

* processing_fee

* gst

* net_disbursement

* loan_tenure_months

* principal_outstanding

* bank_balance_at_application

* disbursal_date

* installment_start_dt

* default (Indicates whether the loan was defaulted)

# Credit Bureau Data (df_bureau)

Contains credit history and financial behavior of customers.

Columns:

* cust_id (Customer ID)

* number_of_open_accounts

* number_of_closed_accounts

* total_loan_months

* delinquent_months

* total_dpd (Days past due)

* enquiry_count

* credit_utilization_ratio

# Importance of Train-Test Split Before EDA and Data Cleaning
Before performing data cleaning and exploratory data analysis (EDA), we split the dataset into training and testing sets to prevent data leakage and ensure a fair evaluation of the model. Data leakage occurs when information from the test set influences the analysis or feature engineering process, leading to overly optimistic model performance that may not generalize well to unseen data. This issue, known as train-test contamination, can cause misleading insights and unrealistic expectations about the model’s effectiveness. By keeping the test set separate and untouched until the final evaluation, we ensure that all decisions regarding feature selection, transformations, and model tuning are based solely on the training data. This approach helps create a robust and unbiased model that performs well in real-world scenarios.

# Data Cleaning
* Handling Missing Values
![image](https://github.com/user-attachments/assets/58c55aba-ceed-4b48-b8b2-30730385526d)

The **Residence_Type** column has missing values, which we will fill using the mode (most frequent value) to maintain data consistency and avoid introducing bias. This ensures minimal impact on the dataset while preserving the original distribution.

# Boxplot To Visualize Outliers
![image](https://github.com/user-attachments/assets/a46fe6c8-00a5-4b85-b6c7-b6f71502b557)

* We observed outliers in the **Processing_Fee** column, indicating cases where the fee is higher than the loan amount, which is not possible. These anomalies suggest data errors or inconsistencies that need to be addressed for accurate   analysis.
* We discussed with the business team and decided that if the **Processing_Fee** exceeds **3% of the loan amount**, it will be considered an outlier. This approach helps maintain data integrity and ensures realistic fee values in the     analysis.
* We also found that some values in the **GST** column are outliers, as GST should not exceed **20%**. Any values above this threshold will be treated as anomalies to ensure data accuracy and consistency.

# Analyze Categorical Columns
![image](https://github.com/user-attachments/assets/d5204513-55b0-4146-bd99-b06a573facb9)

There is a spelling mistake in the **Loan_Purpose** column where "Personnal" should be corrected to "Personal." We will clean this to ensure data consistency and accuracy.

# Exploratory Data Analysis
![image](https://github.com/user-attachments/assets/dd43bdb0-1ec1-408d-9d7f-5a0b63e93524)

* Average age in the default group is little less (37.12) than the average (39.7) of the group that did not default
* Variability (standard deviation) is mostly similar in both the groups
* Both the groups have similar min and max ages

![image](https://github.com/user-attachments/assets/fc038f6d-956f-4b23-83a2-96075f06688a)

* The **Orange (defaulted) group** is slightly shifted to the left, indicating that younger individuals are more likely to default on their loans. This trend suggests a possible relationship between age and loan repayment behavior, which could be explored further for risk assessment.

# KDE for all columns
![image](https://github.com/user-attachments/assets/1d346f7b-1cab-4509-b6e2-d606a120f7bf)

* In columns: loan_tenure_months, delinquent_months, total_dpd, credit_utilization, higher values indicate high likelyhood of becoming a default. Hence these 4 looks like strong predictors
* In remaining columns the distributions do not give any obvious insights
* Why loan_amount and income did not give any signs of being strong predictors? May be when we combine these two and get loan to income ratio (LTI), that may have influence on the target variable. We will explore more later

# Feature Engineering, Feature Selection
* Generate Loan to Income (LTI) Ratio
![image](https://github.com/user-attachments/assets/9f5da98f-cc46-457d-baa9-0fc1d9d4e41c)

![image](https://github.com/user-attachments/assets/0771eb90-8f09-4a13-b43d-138f91b051bb)

* Blue graph has majority of its values on lower side of LTI
* Orange graph has many values when LTI is higher indicating that higher LTI means high risk loan

## Generate Delinquency Ratio
![image](https://github.com/user-attachments/assets/7f7953e2-1eba-40c9-a19b-6fb94a7f8d00)

![image](https://github.com/user-attachments/assets/2f973900-bed0-4835-8493-3714b7bd7f29)

* Blue graph has majority of its values on lower side of LTI
* Orange graph has many values when delinquency ratio is higher indicating some correlation on default

## Generate Avg DPD Per Delinquency
![image](https://github.com/user-attachments/assets/b0c4435e-bf20-45e1-bf64-7b5ee76dffe4)

![image](https://github.com/user-attachments/assets/37d22b2f-77cf-47da-bcc5-38371c60db78)

* Graph clearly shows more occurances of default cases when avg_dpd_per_delinquency is high. This means this column is a strong predictor

## Droping Columns
* Remove columns that are just unique ids and don't have influence on target
* Remove columns that business contact person asked us to remove

## VIF to measure multicolinearity
![image](https://github.com/user-attachments/assets/ec1bde5f-09fd-4567-b6a3-223cb3f23107)

* By performing **Variance Inflation Factor (VIF) analysis**, we found that the columns **Sanction_Amount, Processing_Fee, GST, Net_Disbursement, and Principal_Outstanding** have high VIF values, indicating multicollinearity. To avoid
  redundancy and improve model performance, we will drop these columns.

## Feature Selection for Categorical Variables Using WOE and IV
* To select important categorical features, we calculate Weight of Evidence (WOE) and Information Value (IV):
* Weight of Evidence (WOE): It measures the predictive power of a categorical variable by comparing the distribution of good and bad cases (e.g., loan repayment vs. default). Higher WOE values indicate stronger separation between          classes.
* Information Value (IV): It quantifies the importance of a variable in predicting the target outcome. Variables with higher IV values contribute more to the model. Generally, IV is interpreted as:
    * < 0.02 → Not useful
    * 0.02 - 0.1 → Weak predictor
    * 0.1 - 0.3 → Medium predictor
    * 0.3 - 0.5 → Strong predictor
    * > 0.5 → Very strong predictor (may indicate overfitting)
* By calculating WOE and IV, we can identify and retain the most relevant categorical features for better model performance.

![image](https://github.com/user-attachments/assets/377e0ba1-c3cd-4aee-bb3f-a864a72dfd44)

## Feature Encoding
Feature encoding is the process of converting categorical variables into numerical values so that machine learning models can process them effectively. Since most algorithms require numerical input, categorical data must be transformed appropriately. There are several encoding techniques, including **Label Encoding**, which assigns a unique integer to each category, and **One-Hot Encoding**, which creates binary columns for each category to avoid introducing ordinal relationships. For high-cardinality variables, **Target Encoding** and **WOE Encoding** are useful, as they replace categories with meaningful statistics based on the target variable. Choosing the right encoding method ensures that categorical features contribute effectively to the model without introducing bias or redundancy.

# Model Training
## Attempt 1
* Logistic Regression, RandomForest & XGB
* No handling of class imbalance

  
Logistic regression

  ![image](https://github.com/user-attachments/assets/df1e736b-cbb8-444b-93ee-99763d52cf12)

RandomForestClassifier

![image](https://github.com/user-attachments/assets/000a75d0-4bfc-42bb-9501-f58fc737ea77)

XGBoost

![image](https://github.com/user-attachments/assets/65a5376f-14e7-453b-a88d-96a8dde4322d)

Since there is not much difference between XGB and Logistic, we will choose LogisticRegression as a candidate for our RandomizedSearchCV candidate it has a better interpretation.

## Attempt 2
  * Logistic Regression & XGB
  * Handle Class Imbalance Using Under Sampling
    
Undersampling Handling

![image](https://github.com/user-attachments/assets/5de12e07-2e09-4b27-857a-3c77a13ceb3c)

Logistic Regression 

![image](https://github.com/user-attachments/assets/bf85a482-3c68-4a2e-9994-899774c0dd70)

XGBoost

![image](https://github.com/user-attachments/assets/86a914fc-816b-4612-9916-617882824141)

## Attempt 3
  * Logistic Regression
  * Handle Class Imbalance Using SMOTE Tomek
  * Parameter tunning using optuna

Smote

![image](https://github.com/user-attachments/assets/513aa20f-1748-4ada-af06-1032b293a2b3)

Logistic Regression

![image](https://github.com/user-attachments/assets/150ecb3f-3e5e-4ed6-a516-03919a960e5c)

optuna tunning

![image](https://github.com/user-attachments/assets/8f8e621c-ca8c-4cdf-97d5-a2d49ab18f5e)


## Attempt 4
* XGBoost
* Handle Class Imbalance Using SMOTE Tomek
* Parameter tunning using optuna

![image](https://github.com/user-attachments/assets/a039447c-4a46-4bd4-895b-2c74329c06c9)

# Model Evaluation : ROC/AUC

![image](https://github.com/user-attachments/assets/b9105516-db40-4e1d-a938-ac6690d5e61d)

# Model Evaluation : Rankordering, KS statistic, Gini coeff

* **Rank Ordering**: This evaluates how well the model separates good and bad cases by ranking predictions from highest to lowest. A well-performing model should show a clear separation, where higher scores correspond to higher                probabilities of the target event (e.g., loan default).

* **KS Statistic** (Kolmogorov-Smirnov Statistic): This measures the maximum difference between the cumulative distribution of positive and negative classes. A higher KS value (typically between 0 and 1) indicates better model                 discrimination. KS > 0.4 is generally considered good.

* **Gini Coefficient**: This measures inequality in prediction distribution and is derived from the Lorenz Curve. It is calculated as 2 × AUC - 1, where AUC is the Area Under the Curve of the ROC curve. A Gini coefficient closer to 1          indicates a highly effective model, while a value near 0 suggests poor performance.

  ![image](https://github.com/user-attachments/assets/05b47443-369b-4971-832e-f39481d38227)
  
## Insights from the Decile Table

  * Top Deciles
  * The first decile (Decile 9) has a high event rate of 72.00% and a non-event rate of 28.00%. This indicates that the model is highly confident in predicting events in this decile.
  * The second decile (Decile 8) also shows a significant event rate of 12.72%, with a cumulative event rate reaching 98.6%.
  * Middle Deciles:
  * Deciles 7 and 6 show a significant drop in event rates
  * Lower Deciles:
  * Deciles 5 to 0 show zero events, with all predictions being non-events. These deciles collectively have a non-event rate of 100%.
  * KS Statistic:
  * The KS statistic, which is the maximum difference between cumulative event rates and cumulative non-event rates, is highest at Decile 8 with a value of 85.98%. This suggests that the model performs best at distinguishing between         events and non-events up to this decile.The KS value gradually decreases in the following deciles, indicating a decrease in model performance for distinguishing between events and non-events.

### KS Value
* The highest KS value is 85.98%, found at Decile 8. This indicates that the model's performance in distinguishing between events and non-events is most significant at this decile. (If KS is in top 3 decile and score above 40, it is        considered a good predictive model.)

![image](https://github.com/user-attachments/assets/396302e6-84ac-4772-a48d-21f6207b1e84)

* AUC of 0.98: The model is very good at distinguishing between events and non-events.
* Gini coefficient of 0.96: This further confirms that the model is highly effective in its predictions, with almost perfect rank ordering capability.
* The Gini coefficient ranges from -1 to 1, where a value closer to 1 signifies a perfect model, 0 indicates a model with no discriminative power, and -1 signifies a perfectly incorrect model.


















