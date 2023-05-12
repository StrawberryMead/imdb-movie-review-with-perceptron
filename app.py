import os
import numpy as np
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

def activation(x):
    return 1 / (1 + np.exp(-x))

def activation_derivative(x):
    return activation(x) * (1 - activation(x))

def calculate_loss(output, target):
    return (output - target) ** 2

def calculate_accuracy(predictions, targets):
    correct_predictions = [1 if p == t else 0 for (p, t) in zip(predictions, targets)]
    accuracy = sum(correct_predictions) / len(targets)
    return accuracy

def train(text, alpha=0.01, num_epochs=10):
    global epochs
    
    x = vectorizer.transform([text])
    w = model.coef_[0]
    b = model.intercept_[0]
    
    epochs = []
    
    for epoch in range(num_epochs):
        # Forward pass
        net = x.dot(w) + b
        output = activation(net)
        target = 1 if model.classes_[0] == 'pos' else -1
        loss = calculate_loss(output, target)
        accuracy = calculate_accuracy(model.predict(x), [target])

        # Backward pass
        d_loss_d_output = output - target
        d_output_d_net = activation_derivative(net)
        d_loss_d_net = d_loss_d_output * d_output_d_net
        d_net_d_w = x.toarray()[0]
        d_net_d_b = 1

        d_loss_d_w = d_loss_d_net * d_net_d_w
        d_loss_d_b = d_loss_d_net * d_net_d_b

        # Update weights and bias
        w = w - alpha * d_loss_d_w
        b = b - alpha * d_loss_d_b

        # Store epoch information
        epoch_info = {
            'epoch': epoch+1,
            'input': x.toarray()[0],
            'weights': np.round(w, 4),
            'target': target,
            'bias': np.round(b[0], 4),
            'net': net[0],
            'output': np.round(output[0], 4),
            'loss': np.round(loss, 4),
            'accuracy': np.round(accuracy, 4)
        }
        epochs.append(epoch_info)

    return epochs

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        epochs = train(text)
        prediction = model.predict(vectorizer.transform([text]))[0]
        prediction = "Positive" if prediction == "pos" else "Negative"
        return render_template('index.html', prediction=prediction, text=text, epochs=epochs)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
