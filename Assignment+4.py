
# coding: utf-8


# ## Assignment 4 - Understanding and Predicting Property Maintenance Fines
# 
# This assignment is based on a data challenge from the Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)). 
# 
# The Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)) and the Michigan Student Symposium for Interdisciplinary Statistical Sciences ([MSSISS](https://sites.lsa.umich.edu/mssiss/)) have partnered with the City of Detroit to help solve one of the most pressing problems facing Detroit - blight. [Blight violations](http://www.detroitmi.gov/How-Do-I/Report/Blight-Complaint-FAQs) are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket compliance?
# 
# The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. This is where predictive modeling comes in. For this assignment, your task is to predict whether a given blight ticket will be paid on time.
# 
# All data for this assignment has been provided to us through the [Detroit Open Data Portal](https://data.detroitmi.gov/). **Only the data already included in your Coursera directory can be used for training the model for this assignment.** Nonetheless, we encourage you to look into data from other Detroit datasets to help inform feature creation and model selection. We recommend taking a look at the following related datasets:
# 
# * [Building Permits](https://data.detroitmi.gov/Property-Parcels/Building-Permits/xw2a-a7tf)
# * [Trades Permits](https://data.detroitmi.gov/Property-Parcels/Trades-Permits/635b-dsgv)
# * [Improve Detroit: Submitted Issues](https://data.detroitmi.gov/Government/Improve-Detroit-Submitted-Issues/fwz3-w3yn)
# * [DPD: Citizen Complaints](https://data.detroitmi.gov/Public-Safety/DPD-Citizen-Complaints-2016/kahe-efs3)
# * [Parcel Map](https://data.detroitmi.gov/Property-Parcels/Parcel-Map/fxkw-udwf)
# 
# ___
# 
# We provide you with two data files for use in training and validating your models: train.csv and test.csv. Each row in these two files corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. The target variable is compliance, which is True if the ticket was paid early, on time, or within one month of the hearing data, False if the ticket was paid after the hearing date or not at all, and Null if the violator was found not responsible. Compliance, as well as a handful of other variables that will not be available at test-time, are only included in train.csv.
# 
# Note: All tickets where the violators were found not responsible are not considered during evaluation. They are included in the training set as an additional source of data for visualization, and to enable unsupervised and semi-supervised approaches. However, they are not included in the test set.
# 
# <br>
# 
# **File descriptions** (Use only this data for training your model!)
# 
#     train.csv - the training set (all tickets issued 2004-2011)
#     test.csv - the test set (all tickets issued 2012-2016)
#     addresses.csv & latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 
#      Note: misspelled addresses may be incorrectly geolocated.

# 

# 
# 
# ___
# 
# ## Evaluation
# 
# Your predictions will be given as the probability that the corresponding blight ticket will be paid on time.
# 
# The evaluation metric for this assignment is the Area Under the ROC Curve (AUC). 
# 
# Your grade will be based on the AUC score computed for your classifier. A model which with an AUROC of 0.7 passes this assignment, over 0.75 will recieve full points.
# ___
# 
# For this assignment, create a function that trains a model to predict blight ticket compliance in Detroit using `train.csv`. Using this model, return a series of length 61001 with the data being the probability that each corresponding ticket from `test.csv` will be paid, and the index being the ticket_id.


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,auc
from sklearn.linear_model import LogisticRegression


def blight_model():
    
    x = pd.read_csv("train.csv", usecols = [0,1,14,18,20,21,22,23,24,25,33], encoding = 'ISO-8859-1')
    m = pd.read_csv("addresses.csv")
    n = pd.read_csv("latlons.csv")
    mn = m.merge(n, left_on = "address", right_on = "address", how = "inner")[["ticket_id", "lat", "lon"]]
    X = x.merge(mn, left_on = "ticket_id", right_on = "ticket_id", how = "inner")
    X['ticket_issued_date'] = pd.to_datetime(X['ticket_issued_date']).dt.date
    tes = pd.read_csv("test.csv", usecols = [0, 1, 14, 18, 20, 21, 22, 23, 24, 25], encoding = "latin1", low_memory = False)
    test = tes.merge(mn, left_on = "ticket_id", right_on = "ticket_id", how = 'inner')
    test['ticket_issued_date'] = pd.to_datetime(test['ticket_issued_date']).dt.date
    np.random.seed(seed = 0)

    test["agency_name"] = test["agency_name"].fillna(test["agency_name"].mode())
    test["ticket_issued_date"] = test["ticket_issued_date"].fillna(test["ticket_issued_date"].mode())
    test["disposition"] = test["disposition"].fillna(test["disposition"].mode())
    test["admin_fee"] = test["admin_fee"].fillna(test["admin_fee"].median())
    test["state_fee"] = test["state_fee"].fillna(test["state_fee"].median())
    test["late_fee"] = test["late_fee"].fillna(test["late_fee"].median())
    test["discount_amount"] = test["discount_amount"].fillna(test["discount_amount"].median())
    test["clean_up_cost"] = test["clean_up_cost"].fillna(test["clean_up_cost"].median())
    test["judgment_amount"] = test["judgment_amount"].fillna(test["judgment_amount"].median())
    test["lat"] = test["lat"].fillna(test["lon"].median())
    test["lon"] = test["lon"].fillna(test["lon"].median())
    
    X["agency_name"] = X["agency_name"].fillna(X["agency_name"].mode())
    X["ticket_issued_date"] = X["ticket_issued_date"].fillna(X["ticket_issued_date"].mode())
    X["disposition"] = X["disposition"].fillna(X["disposition"].mode())
    X["admin_fee"] = X["admin_fee"].fillna(X["admin_fee"].median())
    X["state_fee"] = X["state_fee"].fillna(X["state_fee"].median())
    X["late_fee"] = X["late_fee"].fillna(X["late_fee"].median())
    X["discount_amount"] = X["discount_amount"].fillna(X["discount_amount"].median())
    X["clean_up_cost"] = X["clean_up_cost"].fillna(X["clean_up_cost"].median())
    X["judgment_amount"] = X["judgment_amount"].fillna(X["judgment_amount"].median())
    X["lat"] = X["lat"].fillna(X["lat"].median())
    X["lon"] = X["lon"].fillna(X["lon"].median())
    
    ucats = set(X["agency_name"])|{"<unknown>"}
    X['agency_name'] = pd.Categorical(X['agency_name'], categories = ucats).fillna('<unknown>').codes
    test['agency_name'] = pd.Categorical(test['agency_name'], categories=ucats).fillna('<unknown>').codes
    ucats = set(X["ticket_issued_date"])|{"<unknown>"}
    X['ticket_issued_date'] = pd.Categorical(X['ticket_issued_date'], categories=ucats).fillna('<unknown>').codes
    test['ticket_issued_date'] = pd.Categorical(test['ticket_issued_date'], categories=ucats).fillna('<unknown>').codes
    ucats = set(X["disposition"])|{"<unknown>"}
    X['disposition'] = pd.Categorical(X['disposition'],categories = ucats).fillna('<unknown>').codes
    test['disposition'] = pd.Categorical(test['disposition'],categories = ucats).fillna('<unknown>').codes
    
    
    X = X.dropna()
    y = X["compliance"]
    X = X.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
    
    
    clf = RandomForestClassifier(random_state = 0, max_features = 6, n_estimators = 150)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    test["compliance"] = clf.predict_proba(test)[:, 1]
    test = test.set_index("ticket_id")    
    return test["compliance"]



blight_model()
