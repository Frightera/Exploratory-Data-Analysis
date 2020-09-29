import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-whitegrid")

from collections import Counter

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

test_PassengerId = test_df["PassengerId"] #storing inital as it will be changed.

###############################################################################################

def bar_plot(variable):
    var = train_df[variable]
    varValue = var.value_counts()
    
    #visualize
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,varValue))
    
category1 = ["Survived", "Sex", "Pclass", "Embarked", "SibSp", "Parch"]
for c in category1:
    bar_plot(c)   
    
###############################################################################################

def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(train_df[variable]) # bins parameter can be added.
    plt.xlabel(variable)    
    plt.ylabel("Frequency")
    plt.title("{}○ distribution with hist".format(variable))
    plt.show()
    
numericVar = ["Fare", "Age", "PassengerId"]
for n in numericVar:
    plot_hist(n)

###############################################################################################

#Basic Data Analysis
train_df[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by = "Survived", ascending = False)
train_df[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by = "Survived", ascending = False)
train_df[["SibSp","Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by = "Survived", ascending = False)
train_df[["Parch","Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by = "Survived", ascending = False)

###############################################################################################

#Outlier Analysis
def detect_outliers(df,features):
    outlier_indices = []
    for n in features:
       Q1 = np.percentile(df[n],25)
       Q3 = np.percentile(df[n],75)
       IQR = Q3-Q1
       outlier_step = IQR * 1.5
       outlier_list_col = df[(df[n]< Q1 - outlier_step) | (df[n]> Q3 + outlier_step)].index
       outlier_indices.extend(outlier_list_col) #store indices
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v>2)
    
    return multiple_outliers

outliers = train_df.loc[detect_outliers(train_df, ["Age", "SibSp", "Parch", "Fare"])]
train_df = train_df.drop(detect_outliers(train_df,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)

###############################################################################################

#Missing Value
train_df_len = len(train_df)
train_df = pd.concat([train_df,test_df],axis = 0).reset_index(drop = True)

train_df.columns[train_df.isnull().any()] #checking
train_df.isnull().sum() # how many of them?

#Fill Fare & Embarked
train_df.boxplot(column="Fare",by = "Embarked")
plt.show()
train_df["Embarked"] = train_df["Embarked"].fillna("C")
train_df[train_df["Embarked"].isnull()]

train_df[train_df["Fare"].isnull()]
train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"] == 3]["Fare"]))

#Fill Age
index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)
for i in index_nan_age:
    age_pred = train_df["Age"][((train_df["SibSp"] == train_df.iloc[i]["SibSp"]) &(train_df["Parch"] == train_df.iloc[i]["Parch"])& (train_df["Pclass"] == train_df.iloc[i]["Pclass"]))].median()
    age_med = train_df["Age"].median()
    if not np.isnan(age_pred):
        train_df["Age"].iloc[i] = age_pred
    else:
        train_df["Age"].iloc[i] = age_med
        
train_df[train_df["Age"].isnull()]