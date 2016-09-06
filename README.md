
# Machine Learning Engineer Nanodegree
## Supervised Learning
## Project 2: Building a Student Intervention System

Welcome to the second project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been provided for you, and it will be your job to implement the additional functionality necessary to successfully complete this project. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a `'TODO'` statement. Please be sure to read the instructions carefully!

In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  

>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

### Question 1 - Classification vs. Regression
*Your goal for this project is to identify students who might need early intervention before they fail to graduate. Which type of supervised learning problem is this, classification or regression? Why?*

**Answer: ** A classification problem is solved by placing an object into one of a finite set of classes. A regression problem is solved by predicting a real-valued output based on input parameters. This is a classification problem because the goal is to place each student into one of two classes:
1. A student needs early intervention
2. A student does not need early intervention


## Exploring the Data
Run the code cell below to load necessary Python libraries and load the student data. Note that the last column from this dataset, `'passed'`, will be our target label (whether the student graduated or didn't graduate). All other columns are features about each student.


```python
# Import libraries
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score

# Read student data
student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"
```

    Student data read successfully!


### Implementation: Data Exploration
Let's begin by investigating the dataset to determine how many students we have information on, and learn about the graduation rate among these students. In the code cell below, you will need to compute the following:
- The total number of students, `n_students`.
- The total number of features for each student, `n_features`.
- The number of those students who passed, `n_passed`.
- The number of those students who failed, `n_failed`.
- The graduation rate of the class, `grad_rate`, in percent (%).



```python
# TODO: Calculate number of students
n_students = student_data.shape[0]

# TODO: Calculate number of features
n_features = student_data.columns.shape[0] - 1

# TODO: Calculate passing students
n_passed = student_data[student_data.passed == "yes"].shape[0]

# TODO: Calculate failing students
n_failed = student_data[student_data.passed == "no"].shape[0]

# TODO: Calculate graduation rate
grad_rate = n_passed * 100.0 / n_students

# Print the results
print "Total number of students: {}".format(n_students)
print "Number of features: {}".format(n_features)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)
```

    Total number of students: 395
    Number of features: 30
    Number of students who passed: 265
    Number of students who failed: 130
    Graduation rate of the class: 67.09%


## Preparing the Data
In this section, we will prepare the data for modeling, training and testing.

### Identify feature and target columns
It is often the case that the data you obtain contains non-numeric features. This can be a problem, as most machine learning algorithms expect numeric data to perform computations with.

Run the code cell below to separate the student data into feature and target columns to see if any features are non-numeric.


```python
# Extract feature columns
feature_cols = list(student_data.columns[:-1])

# Extract target column 'passed'
target_col = student_data.columns[-1] 

# Show the list of columns
print "Feature columns:\n{}".format(feature_cols)
print "\nTarget column: {}".format(target_col)

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

# Show the feature information by printing the first five rows
print "\nFeature values:"
print X_all.head()
```

    Feature columns:
    ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
    
    Target column: passed
    
    Feature values:
      school sex  age address famsize Pstatus  Medu  Fedu     Mjob      Fjob  \
    0     GP   F   18       U     GT3       A     4     4  at_home   teacher   
    1     GP   F   17       U     GT3       T     1     1  at_home     other   
    2     GP   F   15       U     LE3       T     1     1  at_home     other   
    3     GP   F   15       U     GT3       T     4     2   health  services   
    4     GP   F   16       U     GT3       T     3     3    other     other   
    
        ...    higher internet  romantic  famrel  freetime goout Dalc Walc health  \
    0   ...       yes       no        no       4         3     4    1    1      3   
    1   ...       yes      yes        no       5         3     3    1    1      3   
    2   ...       yes      yes        no       4         3     2    2    3      3   
    3   ...       yes      yes       yes       3         2     2    1    1      5   
    4   ...       yes       no        no       4         3     2    1    2      5   
    
      absences  
    0        6  
    1        4  
    2       10  
    3        2  
    4        4  
    
    [5 rows x 30 columns]


### Preprocess Feature Columns

As you can see, there are several non-numeric columns that need to be converted! Many of them are simply `yes`/`no`, e.g. `internet`. These can be reasonably converted into `1`/`0` (binary) values.

Other columns, like `Mjob` and `Fjob`, have more than two values, and are known as _categorical variables_. The recommended way to handle such a column is to create as many columns as possible values (e.g. `Fjob_teacher`, `Fjob_other`, `Fjob_services`, etc.), and assign a `1` to one of them and `0` to all others.

These generated columns are sometimes called _dummy variables_, and we will use the [`pandas.get_dummies()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies) function to perform this transformation. Run the code cell below to perform the preprocessing routine discussed in this section.


```python
def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        
        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

X_all = preprocess_features(X_all)
print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))
```

    Processed feature columns (48 total features):
    ['school_GP', 'school_MS', 'sex_F', 'sex_M', 'age', 'address_R', 'address_U', 'famsize_GT3', 'famsize_LE3', 'Pstatus_A', 'Pstatus_T', 'Medu', 'Fedu', 'Mjob_at_home', 'Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher', 'Fjob_at_home', 'Fjob_health', 'Fjob_other', 'Fjob_services', 'Fjob_teacher', 'reason_course', 'reason_home', 'reason_other', 'reason_reputation', 'guardian_father', 'guardian_mother', 'guardian_other', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']


### Implementation: Training and Testing Data Split
So far, we have converted all _categorical_ features into numeric values. For the next step, we split the data (both features and corresponding labels) into training and test sets. In the following code cell below, you will need to implement the following:
- Randomly shuffle and split the data (`X_all`, `y_all`) into training and testing subsets.
  - Use 300 training points (approximately 75%) and 95 testing points (approximately 25%).
  - Set a `random_state` for the function(s) you use, if provided.
  - Store the results in `X_train`, `X_test`, `y_train`, and `y_test`.


```python
# TODO: Import any additional functionality you may need here
from sklearn.cross_validation import train_test_split
# TODO: Set the number of training points
num_train = 300

# Set the number of testing points
num_test = X_all.shape[0] - num_train

# TODO: Shuffle and split the dataset into the number of training and testing points above
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=42)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])
```

    Training set has 300 samples.
    Testing set has 95 samples.


## Training and Evaluating Models
In this section, you will choose 3 supervised learning models that are appropriate for this problem and available in `scikit-learn`. You will first discuss the reasoning behind choosing these three models by considering what you know about the data and each model's strengths and weaknesses. You will then fit the model to varying sizes of training data (100 data points, 200 data points, and 300 data points) and measure the F<sub>1</sub> score. You will need to produce three tables (one for each model) that shows the training set size, training time, prediction time, F<sub>1</sub> score on the training set, and F<sub>1</sub> score on the testing set.

**The following supervised learning models are currently available in** [`scikit-learn`](http://scikit-learn.org/stable/supervised_learning.html) **that you may choose from:**
- Gaussian Naive Bayes (GaussianNB)
- Decision Trees
- Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
- K-Nearest Neighbors (KNeighbors)
- Stochastic Gradient Descent (SGDC)
- Support Vector Machines (SVM)
- Logistic Regression

### Question 2 - Model Application
*List three supervised learning models that are appropriate for this problem. For each model chosen*
- Describe one real-world application in industry where the model can be applied. *(You may need to do a small bit of research for this — give references!)* 
- What are the strengths of the model; when does it perform well? 
- What are the weaknesses of the model; when does it perform poorly?
- What makes this model a good candidate for the problem, given what you know about the data?

**Answer: **

1. Gaussian Naive Bayes
    - Email Spam Filtering using the "Bag of Words" approach: a count is generated for each word in an email, and those word counts are associated with either legitimate or spam email. https://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering
    - The model "Naively" assumes that each parameter is independent. Because of this independence, they do not suffer greatly from the "curse of dimensionality", and scale well to large numbers of features. They also do not require much training data to begin making good predictions. The model performs well as long as the assumptions of independence hold true.
    - The model's independence assumptions are also its main weakness. If the model's features are too interdependent, the model will do a bad job of capturing the complexity added by those interdependencies. 
    - This model is a good candidate because of it's general effectiveness as a classifier, it's low computational requirements, and its low training data requirements. 


2. Support Vector Machines (SVM)
    - SVMs have been used extensively in the biological sciences for gene classificaion and to detect patterns in biological sequences.
    https://noble.gs.washington.edu/papers/noble_support.pdf
    - Support vector machines maximize the margin between the decision boundary and the "Support" data, can be solved exactly (when solvable), perform well on high-dimensional input, and requires relatively small amounts of input data to make useful predictions.
    - Support vector machines can only directly solve two-class problems, so multi-class problems must be broken down into two-class problems in order for a solution to be found. The scientist may need to try several different kernels in order to find one which effectively splits the data in a way that minimizes bias.
    - This model is a good candidate because our problem is a 2-class problem, has a high ratio of features to training samples, and has low computational requirements.
    

3. Logistic Regression
    - Logistic regression was used to develop the Trauma and Injury Severity Score, which is used heavily to predict the mortality rate of injured patients. https://en.wikipedia.org/wiki/Logistic_regression#Fields_and_example_applications
    - Logistic regression treats input data as variables of a mathematical function, making it easy to reason about (important features have larger coefficients). It has low computational requirements for predictions, and performs reasonably well on both small and large datasets. 
    - Logistic regression generally is only used to solve two-class problems (including one-vs-rest). Models can have high bias and may require larger datasets to stabilize predictions. Models can also be slow to train on small datasets relative to other methods.
    - This model is a good candidate because of it's general effectiveness as a classifier, because our problem is a 2-class problem, and because it has low computational requirements.

### Setup
Run the code cell below to initialize three helper functions which you can use for training and testing the three supervised learning models you've chosen above. The functions are as follows:
- `train_classifier` - takes as input a classifier and training data and fits the classifier to the data.
- `predict_labels` - takes as input a fit classifier, features, and a target labeling and makes predictions using the F<sub>1</sub> score.
- `train_predict` - takes as input a classifier, and the training and testing data, and performs `train_clasifier` and `predict_labels`.
 - This function will report the F<sub>1</sub> score for both the training and testing data separately.


```python
def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print "Trained model in {:.4f} seconds".format(end - start)

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    # Print and return results
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))
```

### Implementation: Model Performance Metrics
With the predefined functions above, you will now import the three supervised learning models of your choice and run the `train_predict` function for each one. Remember that you will need to train and predict on each classifier for three different training set sizes: 100, 200, and 300. Hence, you should expect to have 9 different outputs below — 3 for each model using the varying training set sizes. In the following code cell, you will need to implement the following:
- Import the three supervised learning models you've discussed in the previous section.
- Initialize the three models and store them in `clf_A`, `clf_B`, and `clf_C`.
 - Use a `random_state` for each model you use, if provided.
 - **Note:** Use the default settings for each model — you will tune one specific model in a later section.
- Create the different training set sizes to be used to train each model.
 - *Do not reshuffle and resplit the data! The new training points should be drawn from `X_train` and `y_train`.*
- Fit each model with each training set size and make predictions on the test set (9 in total).  
**Note:** Three tables are provided after the following code cell which can be used to store your results.


```python
# TODO: Import the three supervised learning models from sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# TODO: Initialize the three models
clf_A = GaussianNB()
clf_B = SVC(random_state=42)
clf_C = LogisticRegression(random_state=42)

# TODO: Set up the training set sizes
X_train_100 = X_train[0:100]
y_train_100 = y_train[0:100]

X_train_200 = X_train[0:200]
y_train_200 = y_train[0:200]

X_train_300 = X_train[0:300]
y_train_300 = y_train[0:300]

# TODO: Execute the 'train_predict' function for each classifier and each training set size
train_predict(clf_A, X_train_100, y_train_100, X_test, y_test)
train_predict(clf_A, X_train_200, y_train_200, X_test, y_test)
train_predict(clf_A, X_train_300, y_train_300, X_test, y_test)
print("\n")
train_predict(clf_B, X_train_100, y_train_100, X_test, y_test)
train_predict(clf_B, X_train_200, y_train_200, X_test, y_test)
train_predict(clf_B, X_train_300, y_train_300, X_test, y_test)
print("\n")
train_predict(clf_C, X_train_100, y_train_100, X_test, y_test)
train_predict(clf_C, X_train_200, y_train_200, X_test, y_test)
train_predict(clf_C, X_train_300, y_train_300, X_test, y_test)
print("\n")
```

    Training a GaussianNB using a training set size of 100. . .
    Trained model in 0.0012 seconds
    Made predictions in 0.0005 seconds.
    F1 score for training set: 0.8467.
    Made predictions in 0.0003 seconds.
    F1 score for test set: 0.8029.
    Training a GaussianNB using a training set size of 200. . .
    Trained model in 0.0008 seconds
    Made predictions in 0.0003 seconds.
    F1 score for training set: 0.8406.
    Made predictions in 0.0003 seconds.
    F1 score for test set: 0.7244.
    Training a GaussianNB using a training set size of 300. . .
    Trained model in 0.0008 seconds
    Made predictions in 0.0004 seconds.
    F1 score for training set: 0.8038.
    Made predictions in 0.0003 seconds.
    F1 score for test set: 0.7634.
    
    
    Training a SVC using a training set size of 100. . .
    Trained model in 0.0012 seconds
    Made predictions in 0.0010 seconds.
    F1 score for training set: 0.8777.
    Made predictions in 0.0008 seconds.
    F1 score for test set: 0.7746.
    Training a SVC using a training set size of 200. . .
    Trained model in 0.0031 seconds
    Made predictions in 0.0023 seconds.
    F1 score for training set: 0.8679.
    Made predictions in 0.0012 seconds.
    F1 score for test set: 0.7815.
    Training a SVC using a training set size of 300. . .
    Trained model in 0.0068 seconds
    Made predictions in 0.0052 seconds.
    F1 score for training set: 0.8761.
    Made predictions in 0.0016 seconds.
    F1 score for test set: 0.7838.
    
    
    Training a LogisticRegression using a training set size of 100. . .
    Trained model in 0.0011 seconds
    Made predictions in 0.0002 seconds.
    F1 score for training set: 0.8593.
    Made predictions in 0.0002 seconds.
    F1 score for test set: 0.7647.
    Training a LogisticRegression using a training set size of 200. . .
    Trained model in 0.0024 seconds
    Made predictions in 0.0003 seconds.
    F1 score for training set: 0.8562.
    Made predictions in 0.0004 seconds.
    F1 score for test set: 0.7914.
    Training a LogisticRegression using a training set size of 300. . .
    Trained model in 0.0036 seconds
    Made predictions in 0.0003 seconds.
    F1 score for training set: 0.8468.
    Made predictions in 0.0002 seconds.
    F1 score for test set: 0.8060.
    
    


### Tabular Results
Edit the cell below to see how a table can be designed in [Markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#tables). You can record your results from above in the tables provided.

** Classifer 1 - Gaussian Naive Bayes**  

| Training Set Size | Training Time | Prediction Time (test) | F1 Score (train) | F1 Score (test) |
| :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |
| 100               | 0.0015                  | 0.0003                 | 0.8467           | 0.8029          |
| 200               | 0.0008                  | 0.0003                 | 0.8406           | 0.7244          |
| 300               | 0.0009                  | 0.0003                 | 0.8038           | 0.7634          |

** Classifer 2 - Support Vector Machine**  

| Training Set Size | Training Time | Prediction Time (test) | F1 Score (train) | F1 Score (test) |
| :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |
| 100               | 0.0034                  | 0.0008                 | 0.8777           | 0.7746          |
| 200               | 0.0033                  | 0.0013                 | 0.8679           | 0.7815          |
| 300               | 0.0067                  | 0.0015                 | 0.8761           | 0.7838          |

** Classifer 3 - Logistic Regression**  

| Training Set Size | Training Time | Prediction Time (test) | F1 Score (train) | F1 Score (test) |
| :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |
| 100               | 0.0386                  | 0.0002                 | 0.8593           | 0.7647          |
| 200               | 0.0020                  | 0.0003                 | 0.8562           | 0.7914          |
| 300               | 0.0028                  | 0.0002                 | 0.8468           | 0.8060          |

## Choosing the Best Model
In this final section, you will choose from the three supervised learning models the *best* model to use on the student data. You will then perform a grid search optimization for the model over the entire training set (`X_train` and `y_train`) by tuning at least one parameter to improve upon the untuned model's F<sub>1</sub> score. 

### Question 3 - Choosing the Best Model
*Based on the experiments you performed earlier, in one to two paragraphs, explain to the board of supervisors what single model you chose as the best model. Which model is generally the most appropriate based on the available data, limited resources, cost, and performance?*

**Answer: **

In order to find the best model for predicting student success, three different types of models were chosen as candidates: one Gaussian Naive Bayes Classifier, one Support Vector Machine, and one Logistic Regression Classifier. These three models were selected as candidates due to their fast training and prediction times, low resource requirements, and generally strong performance with little training data. While all three of these models would be able to meet the needs of the school district, the Logistic Regression Classifier is best choice.

The Logistic Regression Classifier is the best choice of the three models because while the model takes longer to train than the other models, it makes predictions faster than any other model with consistently higher accuracy on the test data. If we continue to gather more data, the logistic regression classifier will continue to improve and will perform well even on extremely large datasets. Furthermore, the parameters of the model are easy to reason about, so we can easily learn from the model what factors are most important in determining the success of our students.

### Question 4 - Model in Layman's Terms
*In one to two paragraphs, explain to the board of directors in layman's terms how the final model chosen is supposed to work. Be sure that you are describing the major qualities of the model, such as how the model is trained and how the model makes a prediction. Avoid using advanced mathematical or technical jargon, such as describing equations or discussing the algorithm implementation.*

**Answer: **

The Logistic Regression Classifier is a mathematical function that takes information about a student as input (age, sex, school, etc) and computes the probability that a student will pass their final exam. Each of the student features, like their age, is weighted relative to it's importance as a predictor so that important features will have a larger impact on the final prediction. If the predicted probability is above a certain threshold, we predict that the student will pass. If the probability is below that threshold, we predict that the student won't pass. 

We train the model by letting it practice on the data we already have about our students. For each of the students in our training group, we have the model guess whether or not the student will pass their exam. If it gets the answer wrong, we use a mathematical function to correct our model so that it will hopefully get that answer right next time. We correct the model by either increasing or decreasing the importance of some of the features of our students (make age more important, sex less important, etc) in such a way that the error between our prediction and the correct answer is reduced. The model starts out getting most of the answers wrong, but after we train it for enough time, it's able to make accurate predictions most of the time. 

### Implementation: Model Tuning
Fine tune the chosen model. Use grid search (`GridSearchCV`) with at least one important parameter tuned with at least 3 different values. You will need to use the entire training set for this. In the code cell below, you will need to implement the following:
- Import [`sklearn.grid_search.gridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html) and [`sklearn.metrics.make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).
- Create a dictionary of parameters you wish to tune for the chosen model.
 - Example: `parameters = {'parameter' : [list of values]}`.
- Initialize the classifier you've chosen and store it in `clf`.
- Create the F<sub>1</sub> scoring function using `make_scorer` and store it in `f1_scorer`.
 - Set the `pos_label` parameter to the correct value!
- Perform grid search on the classifier `clf` using `f1_scorer` as the scoring method, and store it in `grid_obj`.
- Fit the grid search object to the training data (`X_train`, `y_train`), and store it in `grid_obj`.


```python
# TODO: Import 'GridSearchCV' and 'make_scorer'
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer

# TODO: Create the parameters list you wish to tune
parameters = {
    'C' : [0.5, 1, 2, 5, 10], 
    'intercept_scaling' : [0.5, 0.8, 1, 1.2, 1.5],
    'tol' : [0.00005, 0.0001, 0.0002, 0.001]}

# TODO: Initialize the classifier
clf = LogisticRegression()

# TODO: Make an f1 scoring function using 'make_scorer' 
f1_scorer = make_scorer(f1_score, pos_label='yes')

# TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = GridSearchCV(clf, parameters, scoring=f1_scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_train, y_train)

# Get the estimator
clf = grid_obj.best_estimator_

# Report the final F1 score for training and testing after parameter tuning
print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train))
print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test))
```

    Made predictions in 0.0003 seconds.
    Tuned model has a training F1 score of 0.8482.
    Made predictions in 0.0003 seconds.
    Tuned model has a testing F1 score of 0.8029.


### Question 5 - Final F<sub>1</sub> Score
*What is the final model's F<sub>1</sub> score for training and testing? How does that score compare to the untuned model?*

**Answer: **

*Untuned*
* F<sub>1</sub> Training: 0.8468
* F<sub>1</sub> Testing: 0.8060

*Tuned*
* Tuned F<sub>1</sub> Training: 0.8482
* Tuned F<sub>1</sub> Testing: 0.8029

The tuned and untuned versions of the Logistic Regression model performed with nearly equivalent accuracy. This was surprising at first, but it makes sense that the default values chosen for the logistic regression model would work well under most conditions. There were fewer parameters that I could modify on the logistic regression model than on, for example, the SVM model, which may mean that I would have been able to have more of an impact on the SVM model through proper tuning. 

> **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
