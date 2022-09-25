# One of the most fundamental tenets of society here in America is the concept of loans and debt. From the national down
# to the individual level, debt is used liberally and often without restraint. Corporations like banks and credit unions
# realize this fact, and as a result, are able to make money off of this system by loaning money to needy individuals at
# an interest rate that generates some kind of ROI for these companies.

# For this project, we imagine that we work at a lending institution and are asked to utilize a dataset to accomplish
# the following goal: can we build a machine learning model that can accurately predict if a borrower will pay off their
# loan in time?

# We first begin by importing the necessary libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

# Because I do most of my programming inside the PyCharm IDE, these simply help with the visualizations of the data
# frames.
pd.set_option('display.max_columns', 80)  # This allows us to view all the column within our PyCharm IDE
pd.set_option('display.width', 1000)  # Formats our rows correctly.
pd.set_option('display.max_rows', 82)

# We then read in our dataframe from our local computer and do some preliminary inspections
path = r'C:\Python\Data Sets\loans_2007.csv'
loans = pd.read_csv(path, dtype='object')
print(loans.head(1))
print("\n")
print(f"There are {len(loans.columns)} columns in our dataframe.")
print("\n")

# A good way to proceed would be to answer the following questions and split our dataset into groups accordingly.
# We should exclude columns/features that:
# disclose information from the future (after the loan has already been funded)
# don't affect a borrower's ability to pay back a loan (e.g. a randomly generated ID value by Lending Club)
# need to be cleaned up and are formatted poorly
# require more data or a lot of processing to turn into a useful feature
# contain redundant information
# As such, we will split our potential features into 3 groups and analyze each group at a time to ensure the features
# we choose to keep match the specified criteria above.

# Upon analysis of our first group of features, we remove the columns below as they are not pertinent to our goal
bad_columns = ['id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'grade', 'sub_grade', 'emp_title', 'issue_d']
loans = loans.drop(bad_columns, axis=1)

# After analyzing the second group of features and reading the corresponding data dictionary, we remove these columns
more_bad_columns = ['zip_code', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp']
loans = loans.drop(more_bad_columns, axis=1)

# Finally, after reading the data dictionary of the last grouping of features, we remove our last batch of bad columns
most_bad_columns = ['total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d',
                    'last_pymnt_amnt']
loans = loans.drop(most_bad_columns, axis=1)

# Let's check how many features now remain in our dataframe
print(f"After applying our criteria to each feature, we have {len(loans.columns)} columns remaining in our dataframe")
print("\n")

# Our next step in this machine learning workflow is to identify which column will be our target. Let's take a look at
# the "loan_status" column in more detail as this appears to be the most promising
print(loans["loan_status"].value_counts())
print("\n")

# Some more research was needed into identifying the definitions of each of these 9 words. Recall that the goal for this
# project is to classify applicants as an "optimal" candidate to pay off loans or as "sub-optimal" (a candidate that'll
# default). As a result, let's remap these elements to transform this column into a binary target column.
loan_status_map = {'Fully Paid': 1, 'Charged Off': 0}

# Let's first remove all rows that do ont contain either fully paid or charged off.
mask = (loans["loan_status"] == 'Fully Paid') | (loans["loan_status"] == 'Charged Off')
updated_loans = loans[mask]

# Let's now apply our mapping to the loan_status column
updated_loans["loan_status"].replace(loan_status_map, inplace=True)
print(updated_loans["loan_status"].value_counts())
print("\n")

# We now wish to remove columns that only contain a single unique value as these "features" wouldn't provide any value
# if implemented into our model
drop_columns = []
for column in updated_loans.columns:  # Loop through each column
    col_series = updated_loans[column].dropna().unique()  # Remove null values and count true unique values
    if len(col_series) == 1:  # If the number of the unique values in the series is only 1
        drop_columns.append(column)  # Append the column name to our list for dropping later

# We now drop the columns that we discovered only had 1 unique value
updated_loans = updated_loans.drop(drop_columns, axis=1)
print(f"After removing these columns, we have {len(updated_loans.columns)} columns remaining in our dataframe")
print("\n")

# ===================================== DATA CLEANING AND FEATURE ENGINEERING ======================================== #
# We are now ready to proceed to the fun part of any Machine Learning project: preparing the features for our model.
# We begin the cleaning process by checking our dataframe for null values. Based upon other factors like whether
# the series is continuous or discrete, or whether it is numerical or categorical will determine what we do next.
null_series = updated_loans.isnull().sum()
COI = null_series.loc[null_series > 0]
print(COI)
print("\n")

# We notice that the emp_length has a lot of null values, but employment history is a big factor in a credit report, so
# we will keep it. Let's perform some analysis on the other columns.
print(updated_loans.pub_rec_bankruptcies.value_counts(normalize=True, ascending=False))
print("\n")

# Ninety-five percent of the values in this column are of one type. This will not provide any useful information when
# it comes time to construct our model. Let's drop this column and remove rows that contain NaN from the rest.
updated_loans = updated_loans.drop('pub_rec_bankruptcies', axis=1)
clean_loans = updated_loans.dropna()

# We also wish to check the data types of the columns for our dataframe. This will help us determine which columns are
# categorical. If the columns are discrete, these columns would be prime candidates for one-hot encoding.
object_loans = clean_loans.select_dtypes(include='object')
print(object_loans.head(1))
print("\n")

# It would appear some columns just need to be recast as floats while others will need to be encoded.
float_columns = ['loan_amnt', 'installment', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc',
                 'pub_rec', 'revol_bal', 'total_acc']
for column in float_columns:
    clean_loans[column] = clean_loans[column].astype('float')

# Let's now extract the categorical columns
categorical_loans = clean_loans.select_dtypes(include='object')
print(categorical_loans.head(1))

# The int_rate and revol_util columns can be recast as floats since they represent percentages. Additionally, the date
# columns will require additional feature engineering. We will drop the date columns for now.
clean_loans['int_rate'] = clean_loans['int_rate'].str.rstrip('%')
clean_loans['int_rate'] = clean_loans['int_rate'].astype('float')
clean_loans['revol_util'] = clean_loans['revol_util'].str.rstrip('%')
clean_loans['revol_util'] = clean_loans['revol_util'].astype('float')

# We will then drop a few extra useless columns
mostest_bad_columns = ['last_credit_pull_d', 'earliest_cr_line', 'addr_state', 'title']
clean_loans = clean_loans.drop(mostest_bad_columns, axis=1)

# We then create a mapping for the emp_length column and replace the text values with numeric ones
emp_len_mapping_dict = {
    "emp_length": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0
    }
}

# We then apply our mapping do the dataframe
clean_loans = clean_loans.replace(emp_len_mapping_dict)

# We'll now dummy encode the remaining categorical columns and concatenate the new columns to our dataframe
cat_columns = ["home_ownership", "verification_status", "purpose", "term"]
dummy_df = pd.get_dummies(clean_loans[cat_columns])
clean_loans = pd.concat([clean_loans, dummy_df], axis=1)
clean_loans = clean_loans.drop(cat_columns, axis=1)

# The next thing that we should consider is the imbalance of our target column. There are 35,000 positive cases, but
# only about 5,000 negative cases. For example, our model could predict one for every single prediction and be correct
# over 85% of the time. We will have to address this issue later in our algorithm

# =========================================== MODEL CONSTRUCTION ===================================================== #
# The first model that we will implement is Logistic Regression. The below code will create the model, and we will
# follow up with performing k_fold cross validation. We will then calculate the precision and recall.
clf = LogisticRegression()
features = clean_loans.drop('loan_status', axis=1)
target = clean_loans["loan_status"]
clf.fit(features, target)
predictions = cross_val_predict(clf, features, target, cv=3)
prediction_series = pd.Series(predictions)

# We now calculate the false positive rate and the true positive rate.
tp_filter = (predictions == 1) & (clean_loans["loan_status"] == 1)
fp_filter = (predictions == 1) & (clean_loans["loan_status"] == 0)
tn_filter = (predictions == 0) & (clean_loans["loan_status"] == 0)
fn_filter = (predictions == 0) & (clean_loans["loan_status"] == 1)
tp = len(predictions[tp_filter])
fp = len(predictions[fp_filter])
fn = len(predictions[fn_filter])
tn = len(predictions[tn_filter])

tpr = tp / (tp + fn)
fpr = fp / (fp + tn)
print(fpr)
print(tpr)

# The fact that our precision and recall are so high indicates that our model is probably just predicting all ones.
# There are two ways to solve this issue: under/over sampling or penalizing wrong predictions of the minority class.
# Let's incorporate the class_weight parameter to attempt to correct this issue
lr = LogisticRegression(class_weight="balanced")
predictions2 = cross_val_predict(lr, features, target, cv=3)
predictions2 = pd.Series(predictions2)
tp_filter = (predictions2 == 1) & (clean_loans["loan_status"] == 1)
fp_filter = (predictions2 == 1) & (clean_loans["loan_status"] == 0)
tn_filter = (predictions2 == 0) & (clean_loans["loan_status"] == 0)
fn_filter = (predictions2 == 0) & (clean_loans["loan_status"] == 1)
tp = len(predictions2[tp_filter])
fp = len(predictions2[fp_filter])
fn = len(predictions2[fn_filter])
tn = len(predictions2[tn_filter])

tpr = tp / (tp + fn)
fpr = fp / (fp + tn)
print("After addressing class imbalance, the Recall equals", fpr)
print("After addressing class imbalance, the Precision equals", tpr)

# The class weight penalty for the above model was around 5.89. Let's manually create our own class weight penalty to
# see if we can fix our model overfitting.
penalty = {0: 10, 1: 1}
clf2 = LogisticRegression(class_weight=penalty)
predictions3 = cross_val_predict(clf2, features, target, cv=3)
predictions3 = pd.Series(predictions3)
tp_filter = (predictions3 == 1) & (clean_loans["loan_status"] == 1)
fp_filter = (predictions3 == 1) & (clean_loans["loan_status"] == 0)
tn_filter = (predictions3 == 0) & (clean_loans["loan_status"] == 0)
fn_filter = (predictions3 == 0) & (clean_loans["loan_status"] == 1)
tp = len(predictions3[tp_filter])
fp = len(predictions3[fp_filter])
fn = len(predictions3[fn_filter])
tn = len(predictions3[tn_filter])

tpr = tp / (tp + fn)
fpr = fp / (fp + tn)
print("After manually adjusting our penalty, the Recall equals", fpr)
print("After manually adjusting our penalty, the Precision equals", tpr)

# Because most lenders will be conservative for who they give money out to, having a low false positive rate will ensure
# our employer that the odds of a sub-par borrower slipping through the cracks and securing a loan from us is very slim.
# However, the tradeoff here is that a lot of potential customers will be missed out. We should always keep this trade-
# off in mind.

# ====================================== RANDOM FOREST CLASSIFIER CONSTRUCTION ======================================= #
# The next model we will try out will be an ensemble model: Random Forest Classifier
rf = RandomForestClassifier(class_weight='balanced', random_state=1)
predictions4 = cross_val_predict(rf, features, target, cv=3)
predictions4 = pd.Series(predictions4)
tp_filter = (predictions4 == 1) & (clean_loans["loan_status"] == 1)
fp_filter = (predictions4 == 1) & (clean_loans["loan_status"] == 0)
tn_filter = (predictions4 == 0) & (clean_loans["loan_status"] == 0)
fn_filter = (predictions4 == 0) & (clean_loans["loan_status"] == 1)
tp = len(predictions4[tp_filter])
fp = len(predictions4[fp_filter])
fn = len(predictions4[fn_filter])
tn = len(predictions4[tn_filter])

tpr = tp / (tp + fn)
fpr = fp / (fp + tn)
print(tpr, fpr)

# ============================================= CONCLUSION =========================================================== #
# The Random Forest Classifier appears to be severely over fitting our dataset. As a result, we wouldn't want to use
# this model due to the fact that our false positive rate is so elevated. Our "best" model was the Logistic Regression
# model that had a false positive rate of about 16%, which is better for a conservative lender.