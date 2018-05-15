# Diabetes-Classification
This project relies on training and evaluating a classification model to predict which hospitalized diabetes patients will be readmitted for their conditions at a later date. To achieve this purpose, we used a dataset that contains records of diabetes patients admitted to US hospitals. Readmission of patients is both a metric of potential poor care as well as a financial burden to patients, insurers, governments and health care providers.
 
 ### Prerequisites
 - An Azure ML account
 - A web browser and Internet connection
 
 Here, we use the dataset provided to categorize the diabetes patients. The steps in this process includes,
 1. Prepare the dataset for analysis
 2. Investigate relationships in the data set with visualization using custom Python code.
 3. Create a two-class logistic classification model.
 4. Evaluate the performance to the classification model.
 
 ### Prepare the Dataset with Python
 First of all, upload the two datasets naming diabetic_data.csv and admissions_mapping.csv. Then the following code creates consistent coding for missing values by mapping any of the multiple missing values codes to `unknown`.
 ```
def prep_admissions(admissions):
import pandas as pd
admissions['admission_type_description'] = ['unknown' if ((x in ['Not Available', 'Not Mapped', 'NULL']) | (pd.isnull(x))) else x
for x in admissions['admission_type_description']]
return admissions
def azureml_main(df):
df = prep_admissions(df)
return df

 ```
 The Azure Machine Learning experiment should look like the following image upto this point. 
 ![alt text](screenshots/cleaningdata.png "")
