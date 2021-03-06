# Introduction
At 2:20 a.m. on April 15, 1912, the British ocean liner Titanic sinks into the North Atlantic Ocean.
The massive ship, which carried 2,224 passengers and crew, got 1502 people killed.

This project will be covering data analysis and by machine learning models predicting the survived people's count.

# Overview of the data
![ex4](https://github.com/Frightera/Exploratory-Data-Analysis/blob/master/Titanic%20Project/images/data%20overview.PNG)
- **PassenderId:** Unique number for each person
- **Survived:** 1 = Yes && 0 = No
- **Pclass:** Passenger class
- **Name:** Name
- **Sex:** Gender of a passenger
- **Age:** Age of a passsenger
- **SibSp:** Number of siblings/spouses
- **Parch:** Number of parents/children
- **Ticket:** Number of the ticket
- **Fare:** Amount of money
- **Cabin:** Cabin types
- **Embarked:** Port where passenger embarked
 
  - ## Basic Data Analysis - Is there a relation?
  - Pclass - Survived ?
  - Sex - Survived ?
  - SibSp _ Survived ?
  - Parch - Survived ?
  ![ex5](https://github.com/Frightera/Exploratory-Data-Analysis/blob/master/Titanic%20Project/images/RelationAnalysis/Correlation.png)
  
  - ## Outlier Detection
  ![ex1](https://github.com/Frightera/Exploratory-Data-Analysis/blob/master/Titanic%20Project/images/outliers.PNG)
  - ## Fill - Missing Values
  ![ex2](https://github.com/Frightera/Exploratory-Data-Analysis/blob/master/Titanic%20Project/images/missing%20values.PNG)
  
  **As we can see people who have embarked from Q, paid less.**
  ![ex3](https://github.com/Frightera/Exploratory-Data-Analysis/blob/master/Titanic%20Project/images/fill_embarked.png)
  
  # Visualization ( [all](https://github.com/Frightera/Exploratory-Data-Analysis/tree/master/Titanic%20Project/images) )
  <img src="https://github.com/Frightera/Exploratory-Data-Analysis/blob/master/Titanic%20Project/images/RelationAnalysis/Parch%20-%20Survived.png" width="250"> | <img src="https://github.com/Frightera/Exploratory-Data-Analysis/blob/master/Titanic%20Project/images/RelationAnalysis/Pclass%20-%20Survived%20-%20Embarked%20-%20Sex%20-%20Fare.png" width="250"> | <img src="https://github.com/Frightera/Exploratory-Data-Analysis/blob/master/Titanic%20Project/images/RelationAnalysis/Pclass%20-%20Survived%20-%20Age%20Under%2018-%20Sex.png" width="250">
  
# Machine Learning Model
- Logistic Regression added.
- SVM
- Ensemble Learning
- Decision Trees
