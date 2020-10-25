from sklearn.model_selection import train_test_split

train_df_len # Out[5]: 881

test = train_df[881:]
test.drop(['Survived'], axis = 1, inplace = True)

train = train_df[0:881]
 
x = train.drop(['Survived'], axis = 1)
y = train['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.32, random_state = 13)

# %% Models
# Logistic Regression

from sklearn.linear_model import LogisticRegression

LogisticReg = LogisticRegression()
LogisticReg.fit(x_train,y_train)
train_score = LogisticReg.score(x_train,y_train)
test_score = LogisticReg.score(x_test,y_test)
print("Train score accuracy: % {}" .format(train_score * 100.0))
print("Test score accuracy: % {}" .format(test_score * 100.0))
"""
Train score accuracy: % 83.13856427378965
Test score accuracy: % 82.62411347517731
"""