# Assignment 2  -  Text classification benchmarks

---
## Introduction and contents
This repository contains two scripts named "log_reg.py" and "nn.py" which are used to fulfill the required tasks. the code is almost identical in the two scripts and the nly difference can be found in which classifcation model is sued. Log_reg.py uses logistic regression to generate the likelihood of an event, in this case if we are dealing with fake or real news. 

## data
The project makes use of the 10,556 news articles in the Fake News Dataset. Each article has a title, a body of content, and a label designating whether the news is true or false. The goal is to correctly identify each article's label based on the text it contains.

## script functions
The log_reg.py and nn.py follow and almost exact course of action:
1. import dependencies, locate the dataset in the "in" folder and split the dataset
2. vectorize the data (vectorization is just converting the input data from its format (text) into vectors that our models can work with)
3. initilize the model, fit it to the data and use the model to predict labels, either a Logistic regression or mlp classifier.
4. print a classifcation report and save it to the "out" folder.
5. save the model and vectorizer to the "model" folder. 

## how to replicate
### copy the repository 
git clone https://github.com/AU-CDS/assignment-2---text-classification-Olihaha

make sure to be in correct directory

(cd assignment-2)

### scripts
Run either setup.sh followed up by run.sh or setupandrun.sh

setup.sh activates a virtual environment, pip installs necessary libraries and deactives.

run.sh activates the virtual environment, runs the scripts and deactives itself again.

runandsetup.sh does both.


## results

Classification report for logistic regression

              precision    recall  f1-score   support

            FAKE       0.79      0.86      0.83       628
            REAL       0.85      0.78      0.81       639

        accuracy                           0.82      1267
       macro avg       0.82      0.82      0.82      1267
    weighted avg       0.82      0.82      0.82      1267

Classification report for neural network

                  precision    recall  f1-score   support

            FAKE       0.82      0.80      0.81       628
            REAL       0.80      0.82      0.81       639

        accuracy                           0.81      1267
       macro avg       0.81      0.81      0.81      1267
    weighted avg       0.81      0.81      0.81      1267

Both our models have very similar peformances meaning that there is not a significant difference between the two. In actuality they both peform very well and the logistic regression peforms only 1% better. given the faster speed of the logistic regression that would be superior for sure.
 