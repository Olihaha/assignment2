import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib

#Load data
df = pd.read_csv("in/fake_or_real_news.csv")
X = df["text"]
Y = df["label"]

#Split  dataset
X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=42)

#Initialize  count vectorizer
vectorizer = CountVectorizer(ngram_range = (1,2),
                             lowercase=True,
                             max_df = 0.95,
                             min_df=0.05,
                             max_features=100)

#Fit and transform training and testing data
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

#start logistic regression and train on data
classifier = MLPClassifier(activation = "logistic",
                           hidden_layer_sizes=(25),
                           max_iter=1000,
                           random_state=42)

classifier.fit(X_train_vect, Y_train)

#Predictions
Y_pred = classifier.predict(X_test_vect)

#Show result
print(classification_report(Y_test, Y_pred))

#save results as csv
nnstats = classification_report(Y_test, Y_pred, output_dict=True)
df = pd.DataFrame(nnstats).transpose()
df.to_csv('out/nnstats.csv')

# save results as txt
report_txt = classification_report(Y_test, Y_pred)
with open('out/nnstats.txt', 'w') as file:
    file.write(report_txt)

#Save models
joblib.dump(vectorizer, 'models/vectorizer.pkl')
joblib.dump(classifier, 'models/nn.pkl')

