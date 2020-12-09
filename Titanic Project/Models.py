from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output

train_df_len # Out[5]: 881

test = train_df[881:]
test.drop(['Survived'], axis = 1, inplace = True)

train = train_df[0:881]
 
x = train.drop(['Survived'], axis = 1)
y = train['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 13)

# %% Some Tradional Models
# Logistic Regression

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

# Support Vector Machines

clf = SVC(gamma='auto')
clf.fit(x_train,y_train)

y_hat = clf.predict(x_test)

train_score = clf.score(x_train,y_train)
test_score = clf.score(x_test,y_test)
print("Train score accuracy: % {}" .format(train_score * 100.0))
print("Test score accuracy: % {}" .format(test_score * 100.0))

def my_custom_loss_func(y_true, y_pred):
     diff = np.abs(y_true - y_pred).max()
     return np.log1p(diff)
 
print('Loss is ' ,(my_custom_loss_func(y_test,y_hat)))
"""
Train score accuracy: % 84.14023372287144
Test score accuracy: % 78.36879432624113 # probably overfitting
Loss is  0.6931471805599453
"""

# Random Forest Classifier
max_depth = 4

max_depth_list_train = []
for i in range(1,15):
    clf = RandomForestClassifier(max_depth = i ,random_state=13)
    clf.fit(x_train, y_train)
    y_hat = clf.predict(x_test)
    
    train_score = clf.score(x_train,y_train)
    test_score = clf.score(x_test,y_test)
    max_depth_list_train.append(train_score)
    print('Test score is %{} at max depth {}' .format(test_score* 100.0, i))
    print('Train score is %{} at max depth {}' .format(train_score* 100.0, i))
    print('\n')
    plt.scatter(i, train_score)
    plt.xlabel('max depth')
    plt.ylabel('train_score')
    plt.title('Max depth and Train Score')
 
# %%  Ensemble Learning
random_state = 13
classifier = [DecisionTreeClassifier(random_state = random_state),
             SVC(random_state = random_state),
             RandomForestClassifier(random_state = random_state),
             LogisticRegression(random_state = random_state),
             KNeighborsClassifier()]

dt_param_grid = {"min_samples_split" : range(10,500,20),
                "max_depth": range(1,20,2)}

svc_param_grid = {"kernel" : ["rbf"],
                 "gamma": [0.001, 0.01, 0.1, 1],
                 "C": [1,10,50,100,200,300,1000]}

rf_param_grid = {"max_features": [1,3,10],
                "min_samples_split":[2,3,10],
                "min_samples_leaf":[1,3,10],
                "bootstrap":[False],
                "n_estimators":[100,300],
                "criterion":["gini"]}

logreg_param_grid = {"C":np.logspace(-3,3,7),
                    "penalty": ["l1","l2"]}

knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),
                 "weights": ["uniform","distance"],
                 "metric":["euclidean","manhattan"]}
classifier_param = [dt_param_grid,
                   svc_param_grid,
                   rf_param_grid,
                   logreg_param_grid,
                   knn_param_grid]

cv_result = []
best_estimators = []
for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)
    clf.fit(x_train,y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    print(cv_result[i])

cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier", "SVM","RandomForestClassifier",
             "LogisticRegression",
             "KNeighborsClassifier"]})

g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)
g.set_xlabel("Mean Accuracy")
g.set_title("Cross Validation Scores")

votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),
                                        ("rfc",best_estimators[2]),
                                        ("lr",best_estimators[3])],
                                        voting = "soft", n_jobs = -1)
votingC = votingC.fit(x_train, y_train)
print(accuracy_score(votingC.predict(x_test),y_test))

# %% What about Deep Learning?
import tensorflow as tf
from tensorflow import keras

"""
if you want to see it in EVERY epoch,
i.e epoch = 128 means 128 plots.-
Of course you can change it by playing the code.
Pass plot to the callbacks in model fit.

class PlotLearning(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        
        clear_output(wait=True)
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()
        
        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()
        
        plt.show();
        
plot = PlotLearning()
"""
# Not the best choices you can set different layers, since training set is small,
# you won't be able to get desired accuracy.
# Also there is a high chance of overfitting.
# I was messing with layers to see what's gonna happen.
model = keras.models.Sequential([
        keras.layers.Dense(40, input_shape = (x_train.shape[1],), activation = 'relu'),
        keras.layers.Dense(32, activation = 'relu'),
        keras.layers.Dense(64, activation = 'relu'),
        keras.layers.Dense(80, activation = 'relu'),
        keras.layers.Dense(64, activation = 'relu'),
        keras.layers.Dense(32, activation = 'relu'),
        keras.layers.Dense(32, activation = 'relu'),
        keras.layers.Dense(1, activation = 'sigmoid')                      
    
                                ])
model.summary()

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adamax


optimizer = RMSprop(lr=0.001)
optimizer2 = Adamax(lr = 0.0001)

epochs = 256
steps_per_epoch = 32
batch_size = 16

model.compile(optimizer=optimizer2,
              loss='binary_crossentropy',
              metrics = ['accuracy'])
history = model.fit(x_train,y_train, epochs = epochs,
                    validation_data=(x_test, y_test),
                    steps_per_epoch = steps_per_epoch,
                    batch_size = batch_size,
                    verbose = 2)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()









    
