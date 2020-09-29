import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-whitegrid")

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

list1 = ["SibSp", "Parch", "Age", "Fare", "Survived"]
sns.heatmap(train_df[list1].corr(), annot = True, fmt = ".2f")
plt.show()
#Age is not correlated with sex but it is correlated with parch, sibsp and pclass.#

#SibSp - Survived
g = sns.catplot(x = "SibSp", y = "Survived", data = train_df, kind = "bar", height= 9)
g.set_ylabels("Probability of Survival"),
plt.show()

#ParCh - Survived
g = sns.catplot(x = "Parch", y = "Survived", data = train_df, kind = "bar", height = 7)
g.set_ylabels("Probability of Survival"),
plt.show()
print("Small families have more chance to survive")

#Plcass - Survived - Sex
n = sns.catplot(x = "Pclass", y = "Survived", hue = "Sex", data = train_df, palette={"male": "g", "female": "m"}, kind = "point", height = 8)
n.set_ylabels("Probability of Survival")
plt.show()

# Age - Survived - Sex
n = sns.FacetGrid(train_df, col = "Survived", row = "Sex")
n.map(sns.distplot, "Age", bins = 20)
plt.show()


#Pclass - Survived - Age - Embarked
n = sns.FacetGrid(train_df, row = "Embarked")
n.map(sns.pointplot, "Age", "Pclass", "Survived")
n.add_legend()
plt.show()

#Pclass - Survived - Age Under 18- Sex
under_18 = train_df[train_df["Age"] < 18 ]
n = sns.FacetGrid(under_18, row = "Sex")
n.map(sns.pointplot, "Age", "Survived", "Pclass")
n.add_legend()
plt.show()

#Pclass - Survived - Embarked - Sex - Fare
n = sns.FacetGrid(train_df, row = "Embarked", col = "Survived")
n.map(sns.barplot, "Sex", "Fare", "Pclass")
n.add_legend()
plt.show()

