from flask import Flask, redirect, render_template, request, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import *
from sklearn.svm import *
import pandas
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

#Create DB table and columns.
class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.Integer)
    prediction = db.Column(db.String(100))
    prediction_probability = db.Column(db.Float)
    error = db.Column(db.String(1064))

global Classifier
global Vectorizer 

# Loading SMS dataset and splitting into training set and test set.
dataset = pandas.read_csv('spam.csv', encoding='latin-1')
# 4400 items.
training_set = dataset[:4400]
# 1172 items.
test_set = dataset[4400:]

# Training model.
Classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True))
Vectorizer = TfidfVectorizer()
vectorize_text = Vectorizer.fit_transform(training_set.v2)
Classifier.fit(vectorize_text, training_set.v1)

# Flask /index route.
@app.route('/')
def index():
    sms_messages = Message.query.all()
    return render_template('index.html', sms_messages=sms_messages)

# Flask /process route, for processing SMS message submissions.
@app.route('/process', methods=['POST'])
def process():
    message = request.form['sms_message']
    prediction = ''
    prediction_probability = ''
    error = ''

    global Classifier
    global Vectorizer

    try:
        if len(message) > 0:
            vectorize_message = Vectorizer.transform([message])
            prediction = Classifier.predict(vectorize_message)[0]
            prediction_probability = Classifier.predict_proba(vectorize_message).tolist()
    except BaseException as instance:
        error = str(type(instance).__name__) + ' ' + str(instance)

    new_message = Message(message=message, prediction=prediction, prediction_probability=prediction_probability[0][0], error=error)
    db.session.add(new_message)
    db.session.commit()

    return redirect(url_for('index'))

if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)