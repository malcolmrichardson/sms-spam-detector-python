from sklearn.naive_bayes import *
from sklearn.dummy import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.svm import *
import pandas

def perform(classifiers, vectorizers, train_data, test_data):
    best_performance_score = 0
    best_combination = ''

    for classifier in classifiers:
        for vectorizer in vectorizers:
            string = ''
            string += classifier.__class__.__name__ + ' with ' + vectorizer.__class__.__name__

            # Training.
            vectorize_text = vectorizer.fit_transform(train_data.v2)
            classifier.fit(vectorize_text, train_data.v1)

            # Scoring.
            vectorize_text = vectorizer.transform(test_data.v2)
            score = classifier.score(vectorize_text, test_data.v1)
            string += ' . Has Score: ' + str(score)
            print(string)

            if score > best_performance_score:
                best_performance_score = score
                best_combination = classifier.__class__.__name__ + ' with ' + vectorizer.__class__.__name__

    print('\nHighest score is ' + best_combination + ' with score of: ' + str(100*best_performance_score) + '%')

# Open Kaggle SMS Spam data set, split it into a training set and a test set.
dataset = pandas.read_csv('spam.csv', encoding='latin-1')
training_set = dataset[:4400]
test_set = dataset[4400:]

perform(
    [
        BernoulliNB(),
        RandomForestClassifier(n_estimators=100, n_jobs=1),
        AdaBoostClassifier(),
        BaggingClassifier(),
        ExtraTreesClassifier(),
        GradientBoostingClassifier(),
        DecisionTreeClassifier(),
        CalibratedClassifierCV(),
        DummyClassifier(),
        PassiveAggressiveClassifier(),
        RidgeClassifier(),
        RidgeClassifierCV(),
        SGDClassifier(),
        OneVsRestClassifier(SVC(kernel='linear')),
        OneVsRestClassifier(LogisticRegression()),
        KNeighborsClassifier()
    ],
    [
        CountVectorizer(),
        TfidfVectorizer(),
        HashingVectorizer(),

    ],
    training_set,
    test_set
)