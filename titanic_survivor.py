# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
""" This is probably not the best way of coding. 
I've dumped everythin into one file isntead og having classes
and calling them from the main. But the idea was to develop a
quick up and running model""" 
  
import pandas as pd
# We need numpy to perform error metrics and some calculations
import numpy as np
# importing Linear regresion from sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# cross validation helper function
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest, f_classif
import re
import operator

titanic = pd.read_csv("/Users/Vineets/Documents/Data Science Material/Kaggle Competitions/titanic_data.csv")

print (titanic.head(5))
print (titanic.describe())

"""
Age has some missing values. We cannot delete the column 
since age is an important variable.Therefore, we fill it 
with the median value.
"""

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

"""
We'll ignore the non numeric colums like Ticket, Cabin and Name since
they wont help us predict the output variable.
We also need to convert the values in Sex and Embarked columns 
to numberic values.
"""

# Set male = 0 and female = 1
titanic.loc[titanic["Sex"] == "male", "Sex" ] = 0
titanic.loc[titanic["Sex"] =="female", "Sex" ] = 1
titanic["Sex"].describe()

# Set nan values in Embarked column to "S" since "S" is most occuring
titanic["Embarked"] = titanic["Embarked"].fillna("S")

# Set "S" = 0, "C" = 1 and "Q" = 2
titanic.loc[titanic["Embarked"] == "S", "Embarked" ] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked" ] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked" ] = 2
titanic["Embarked"].describe()


"""
Predicting the survivor, that is, the "Survived" variable
using the predictor variables.
"""

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare","Embarked"]

#Initialize our algorithm
algo = LinearRegression()

# Generating cross validation folds for the titanic datasets.
# It returns the row indices corresponding to the train and test sets.
# We set random_state so the we get the same splits every time we run this.

kf = KFold(titanic.shape[0], n_folds = 3, random_state = 1)
predictions = []

for train, test in kf:
    #we take the predictors and the target variables to train the algorithm
    train_predictors = (titanic[predictors].iloc[train,:])
    train_target = titanic["Survived"].iloc[train]
    # training the algo. by running linear regression
    algo.fit(train_predictors, train_target)
    
    # Now that the model has been trained, lets test it on the test set
    test_predictions  = algo.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)
    
    
""" Evaluating the model :
Error metric for this dataset is % of correct predictions. 
We need to check the number of correct prediction and
divide it by the total number of passengers. """

# First we concatinate all the folds  into one array
# A value if > 0.5 means class 1 and a value of <= 0.5 means class 0

predictions = np.concatenate(predictions, axis =0)
predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0
print "Accuracy measure"
print "_____________________________"
#Finally lets calculate the accuracy
accuracy = sum(predictions[predictions == titanic["Survived"]])/ len(predictions)
print ("Linear Regression: " + str(accuracy*100))

# We got an accuracy of 78.3%. We can improve it using some other algorithm
# lets try logistic regression

logreg = LogisticRegression(random_state = 1)

# Computing accuracy score is much simpler
scores = cross_validation.cross_val_score(logreg, titanic[predictors], titanic["Survived"], cv=3)
print ("Logistic Regression: " + str(scores.mean()*100))

print "_______________________________"
print "Predicting Survived on test sets."

print "Lets clean data first"

titanic_test = pd.read_csv("/Users/Vineets/Documents/Data Science Material/Kaggle Competitions/test.csv")
titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0 
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")

titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

# Initializing the logistic regression algorithm
alg = LogisticRegression(random_state=1)

# Train the algorithm using all the training data
alg.fit(titanic[predictors], titanic["Survived"])

# Make predictions using the test set.
predictions = alg.predict(titanic_test[predictors])
# print  predictions

print "____________________________"
print "Implementing Random Forest Classifier"

# Initialize our algorithm with the default paramters
# n_estimators is the number of trees we want to make
# min_samples_split is the minimum number of rows we need to make a split
# min_samples_leaf is the minimum number of samples we can have at the place
# where a tree branch ends (the bottom points of the tree)

algo_RF = RandomForestClassifier(random_state = 3, n_estimators = 150, min_samples_split = 4, min_samples_leaf = 2)
scores = cross_validation.cross_val_score(algo_RF, titanic[predictors], titanic["Survived"], cv = 3)
print ("Random Forest: " + str(scores.mean()*100))

submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
        })
submission.to_csv("/Users/Vineets/Documents/Data Science Material/Kaggle Competitions/kaggle.csv", index=False)
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 15:48:59 2015

@author: Vineets
"""


""" To further improve our model we can generate new features
like the length of the name of the person can determine their position
in titanic. Also, total number of people in family = SibSp + Parch
"""
# Generating a family size column
titanic["FamilySize"] = titanic["SibSp"]  + titanic["Parch"]

#Generating a NameLength column
titanic["NameLength"] = titanic["Name"].apply(lambda x :len(x))

# Extracting titles lke Mr., Master, Mrs. from the name using re

#The follwing funcction extracts title from the name
def get_title(name):
    # Extrach titles using regular expression
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # return title if it exists
    if title_search:
        return title_search.group(1)
    return ""
    
# Get all the titles
titles  = titanic["Name"].apply(get_title)
print (pd.value_counts(titles))

# Mapping the titles to an integer. Some titles are very rare and 
# are compressed into same codes as other titles
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, 
                 "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, 
                 "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, 
                 "Countess": 10, "Jonkheer": 10, "Sir": 9, 
                 "Capt": 7, "Ms": 2}
for key,value in title_mapping.items():
    titles[titles == key] = value

# Verify that we converted everything
print (pd.value_counts(titles))

# Finally creating the titles column
titanic["Title"] = titles


"""We can also generate a feature indicating which family people are in. 
Because survival was likely highly dependent on your family and the 
people around you, this has a good chance at being a good feature.
"""
family_id_mapping = {} # A dictionary nmapping name to id

# the following function returns the family_id, given a row

def get_family_id(row):
    # get hte last name using split
    last_name = row["Name"].split(",")[0]
    # Create the family id
    family_id = "{0}{1}".format(last_name,row["FamilySize"])
    # lookup the id in dictioanry
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
           # Get the maximum id from the mapping 
           # and add one to it if we don't have an id
            current_id = (max(family_id_mapping.items(), key = operator.
            itemgetter(1))[1] + 1)
        family_id_mapping[family_id]= current_id
    return family_id_mapping[family_id]
    
# call this function using apply
family_ids = titanic.apply(get_family_id, axis = 1)

# There are a lot of family ids, so we'll compress all of the families under 3 members into one code.
family_ids[titanic["FamilySize"] < 3] = -1

# Print the count of each unique id.
print(pd.value_counts(family_ids))

titanic["FamilyId"] = family_ids

# updating the predictors
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", 
              "Fare", "Embarked", "FamilySize", "Title",
              "FamilyId"]
# selecting the best features
selector = SelectKBest(f_classif, k= 5)
selector.fit(titanic[predictors], titanic["Survived"])

# Get the raw p-value for each feature and convert it to scores
scores =- np.log10(selector.pvalues_)
# print scores

# Plot the scores
"""plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation = 'vertical')
plt.show()
"""
# Pick only the four best features.
predictors = ["Pclass", "Sex", "Fare", "Title"]

algo_RF_new = RandomForestClassifier(random_state = 3, n_estimators = 150, min_samples_split = 4, min_samples_leaf = 4)
scores = cross_validation.cross_val_score(algo_RF_new, titanic[predictors], titanic["Survived"], cv=3)

print ("Random Forest Updated" + str(scores.mean()))
