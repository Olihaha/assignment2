# Assignement 2  -  Text classification benchmarks

---
## Introduction and contents
This repository contains two scripts named "log_reg.py" and "nn.py" which are used to fulfill the required tasks. the code is almost identical in the two scripts and the nly difference can be found in which classifcation model is sued. Log_reg.py uses logistic regression to generate the likelihood of an event, in this case if we are dealing with fake or real news. 
 

## data
The project makes use of the 10,556 news articles in the Fake News Dataset. Each article has a title, a body of content, and a label designating whether the news is true or false. The goal is to correctly identify each article's label based on the text it contains.

## models 
The logistic regression model analyzes a set of features extracted from news articles. These features can include textual characteristics such as word frequencies, sentence structure, grammatical errors, and semantic cues. Logistic regression is a simple linear model that assumes a linear relationship between the features and the output. It uses a logistic or sigmoid function to map the linear combination of the inputs to a probability score. in oversimplified terms, the log reg model defines the data as either 1 or 0 depending on real or fake. it then in easy to visualize terms, graphs out where the current article is in this context and weighs if its closer to a 0 or a 1 and uses this to estimate the result. 

On the other hand, the neural network model consists of multiple layers that interconnect and add different weights to the dataset. These layers include an input layer, one or more hidden layers, and an output layer. Each neuron in a hidden layer applies a weighted sum of inputs, followed by an activation function, to produce an output. The outputs of the neurons in one layer serve as inputs to the neurons in the subsequent layer. 

Neural networks have a significant advantage in the usage of "deep learning" which we are not doing in this assignemnt but will be doing in later assigments. Reading this, one could assume that neural networks are move effective than a logistic regression model, but given that we are working with a smaller dataset and simple labeling, this is not necesarrily the case.

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
 I've saved both the files as csv and txt files. 