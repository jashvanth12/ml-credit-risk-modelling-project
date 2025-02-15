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




