from sklearn.naive_bayes import *
from sklearn.dummy import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.svm import *
import pandas
import csv

# Open Kaggle SMS Spam dataset, split it into a training set and a test set.
dataset = pandas.read_csv('spam.csv', encoding='latin-1')
# 4400 items.
training_set = dataset[:4400]
# 1172 items.
test_set = dataset[4400:]

classifier = OneVsRestClassifier(SVC(kernel='linear'))
vectorizer = TfidfVectorizer()

# Training.
vectorize_text = vectorizer.fit_transform(training_set.v2)
classifier.fit(vectorize_text, training_set.v1)

# Creating .csv array.
csv_arr = []
for index, row in test_set.iterrows():
    solution = row[0]
    text = row[1]
    vectorize_text = vectorizer.transform([text])
    prediction = classifier.predict(vectorize_text)[0]
    if prediction == solution:
        result = 'correct'
    else:
        result = 'incorrect'
    csv_arr.append([len(csv_arr), text, solution, prediction, result])

# Write to .csv.
with open('test_score.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';', quotechar = '"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['#', 'text', 'solution', 'prediction', 'result'])
    for row in csv_arr:
        writer.writerow(row)