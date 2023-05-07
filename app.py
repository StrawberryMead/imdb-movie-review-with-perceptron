import os
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

app = Flask(__name__)

def load_data():
    data = []
    labels = []
    for label in ['pos', 'neg']:
        path = os.path.join('aclImdb', 'train', label)
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), 'r', encoding='utf-8') as file:
                data.append(file.read())
                labels.append(label)
    return data, labels

data, labels = load_data()

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

model = Perceptron()
model.fit(X_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        prediction = model.predict(vectorizer.transform([text]))[0]
        prediction = "Positive" if prediction == "pos" else "Negative"
        return render_template('index.html', prediction=prediction, text=text)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
