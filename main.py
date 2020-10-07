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

# %% FEATURE ENGINEERING

# Name - Title

train_df['Name'].head(15)
# It is not logical to have a relation between name & survival rate.
# Splitting Mr Mrs etc might be useful.

name = train_df['Name']
train_df['Title'] = [i.split(".")[0].split(",")[-1].strip() for i in name]

sns.countplot(x = 'Title', data = train_df)
plt.xticks(rotation=45)
plt.show()

# convert to categorical
train_df["Title"] = train_df["Title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"other")
train_df["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in train_df["Title"]]
train_df["Title"].head(15)

sns.countplot(x="Title", data = train_df)
plt.xticks(rotation = 45)
plt.show()

g = sns.catplot(x = "Title", y = "Survived", hue = "Sex", data = train_df, kind = "point")
g.set_xticklabels(["Master","Mrs","Mr","Other"])
g.set_ylabels("Survival Probability")
plt.show()

train_df.drop(labels = ["Name"], axis = 1, inplace = True) # no need for name column
train_df = pd.get_dummies(train_df,columns=["Title"])
train_df.head()


# Family Size

train_df["Fsize"] = train_df["SibSp"] + train_df["Parch"] + 1
# add 1 if parch and sibsp are zero at the same time, so the smallest family's size consists to 1.

g = sns.catplot(x = "Fsize", y = "Survived", data = train_df, kind = "bar")
g.set_ylabels("SurvivalProb")
plt.show()

train_df["family_size"] = [1 if i < 5 else 0 for i in train_df["Fsize"]]
#check plot

train_df['family_size'].value_counts()
# 1    1227
# 0      72

g = sns.catplot(x = "family_size", y = "Survived", data = train_df, kind = "bar")
g.set_ylabels("SurvivalRate")
plt.show()
# Small families have more chance to survive, as can be seen.

train_df = pd.get_dummies(train_df, columns= ["family_size"])
train_df.head(15)
"""
        PassengerId  Survived Pclass  ... Fsize  family_size_0  family_size_1
0             1       0.0       3  ...     2              0              1
1             2       1.0       1  ...     2              0              1
2             3       1.0       3  ...     1              0              1
3             4       1.0       1  ...     2              0              1
4             5       0.0       3  ...     1              0              1
5             6       0.0       3  ...     1              0              1
6             7       0.0       1  ...     1              0              1
7             8       0.0       3  ...     5              1              0
8             9       1.0       3  ...     3              0              1
9            10       1.0       2  ...     2              0              1
10           11       1.0       3  ...     3              0              1
11           12       1.0       1  ...     1              0              1
12           13       0.0       3  ...     1              0              1
13           14       0.0       3  ...     7              1              0
14           15       0.0       3  ...     1              0              1
"""

# Embarked

sns.countplot(x = "Embarked", data = train_df)
plt.show()

train_df = pd.get_dummies(train_df, columns=["Embarked"])

# Ticket 

train_df["Ticket"].head(15)
"""
0            A/5 21171
1             PC 17599
2     STON/O2. 3101282
3               113803
4               373450
5               330877
6                17463
7               349909
8               347742
9               237736
10             PP 9549
11              113783
12           A/5. 2151
13              347082
14              350406
"""
tickets_mylist = []
for n in list(train_df.Ticket):
    if not n.isdigit():
        tickets_mylist.append(n.replace(".","").replace("/","").strip().split(" ")[0])
    else:
        tickets_mylist.append("x")
train_df["Ticket"] = tickets_mylist

train_df["Ticket"].head(15)
"""
0         A5
1         PC
2     STONO2
3          x
4          x
5          x
6          x
7          x
8          x
9          x
10        PP
11         x
12        A5
13         x
14         x
"""
train_df = pd.get_dummies(train_df, columns= ["Ticket"], prefix = "T")

# Passenger Class & Sex

sns.countplot(x = "Pclass", data = train_df)
plt.show()

train_df["Pclass"] = train_df["Pclass"].astype("category")
train_df = pd.get_dummies(train_df, columns= ["Pclass"])
train_df.head(7)
"""
   PassengerId  Survived     Sex  ...  Pclass_1  Pclass_2  Pclass_3
0            1       0.0    male  ...         0         0         1
1            2       1.0  female  ...         1         0         0
2            3       1.0  female  ...         0         0         1
3            4       1.0  female  ...         1         0         0
4            5       0.0    male  ...         0         0         1
5            6       0.0    male  ...         0         0         1
6            7       0.0    male  ...         1         0         0
"""
train_df["Sex"] = train_df["Sex"].astype("category")
train_df = pd.get_dummies(train_df, columns=["Sex"])

# Drop ID & Cabin

train_df.drop(labels = ["PassengerId", "Cabin"], axis = 1, inplace = True)
