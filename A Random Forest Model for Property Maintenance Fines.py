
# coding: utf-8

 
# This project is based on a data challenge from the Michigan Data Science Team, which have partnered with the City of Detroit 
# to help solve the blight problem. The problem is issued by the city to individuals who allow their properties to remain in a 
# deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year. Many 
# of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know how 
# can we increase blight ticket compliance? My task is to build a model ith an AUROC (the Area Under the ROC Curve) of 0.7 to 
# predict whether a given blight ticket will be paid on time. This is an assignment project in my Machine Learning class. After 
# the function return a predicted value for the test set, the system will calculate the accuracy score and if it is higher than 
# 0.7, the assignment is passed.


# File descriptions
# 
#     train.csv - the training set (all tickets issued 2004-2011)
#     test.csv - the test set (all tickets issued 2012-2016)
#     addresses.csv & latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 


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
