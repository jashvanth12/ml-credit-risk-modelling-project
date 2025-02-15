# credit-risk-modeling

Introduction

Credit risk modelling is a critical aspect of financial risk management, enabling lenders and financial institutions to assess the likelihood of a borrower defaulting on a loan. The primary objective of this project is to develop a predictive model that estimates the probability of credit risk, assigns a credit score, and determines a risk rating for each applicant.

Objective

The goal of this project is to analyze customer data, loan details, and credit bureau information to predict the creditworthiness of individuals. By leveraging machine learning techniques, we aim to build a model that can accurately predict the probability of default, categorize customers based on their credit scores, and provide a comprehensive risk rating system.

Dataset Overview

This project utilizes three key datasets:

Customer Data (df_customers)

Contains demographic and personal details about customers.

Columns:

cust_id (Customer ID)

age

gender

marital_status

employment_status

income

number_of_dependants

residence_type

years_at_current_address

city

state

zipcode

Loan Data (df_loans)

Provides details about the loans sanctioned to customers.

Columns:

loan_id (Loan ID)

cust_id (Customer ID)

loan_purpose

loan_type

sanction_amount

loan_amount

processing_fee

gst

net_disbursement

loan_tenure_months

principal_outstanding

bank_balance_at_application

disbursal_date

installment_start_dt

default (Indicates whether the loan was defaulted)

Credit Bureau Data (df_bureau)

Contains credit history and financial behavior of customers.

Columns:

cust_id (Customer ID)

number_of_open_accounts

number_of_closed_accounts

total_loan_months

delinquent_months

total_dpd (Days past due)

enquiry_count

credit_utilization_ratio
