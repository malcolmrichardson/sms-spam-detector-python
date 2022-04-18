# sms-spam-detector-python
Application used to classify whether an SMS message is likely a spam message or ham (legitimate) message.<br><br>
It tests various classification and vectorization technique combinations to determine which combination works best on the SMS Spam dataset provided from Kaggle.
<br><br>
Run `classifier.py` to view the scores of each of the combinations.
<br><br>
One-Vs-Rest Classification and  Term Frequency-Inverse Document Frequency vectorization were chosen to perform the determinations.
<br><br>
Application makes classifications on SMS messages with a 98.8% accuracy rate, due to the aforementioned ML techniques above.

## Tools used:
- Python
- Flask
- scikit-learn (Machine Learning library)
- pandas (Data Analysis and Manipulation library)
- SQLite
- SQLAlchemy
- HTML
- CSS

## Installation/Setup

Run the below:

```
mkdir sms-spam-detector-python
cd sms-spam-detector-python

git clone https://github.com/malcolmrichardson/sms-spam-detector-python.git

virtualenv venv
source venv/bin/activate

pip install -r requirements.txt

python app.py
```

Application runs on https://127.0.0.1:5000 by default.

Enjoy and thank you!

Sample capture below:
<br><br><br>

