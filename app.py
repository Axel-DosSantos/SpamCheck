from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Charger les modèles
cv = pickle.load(open("models/cv.pkl", 'rb'))
clf = pickle.load(open("models/clf.pkl", 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        email = request.form['email']
        tokenized_email = cv.transform([email])
        prediction = clf.predict(tokenized_email)
        prediction = 'Spam' if prediction[0] == 1 else 'Non Spam'
    return render_template('form.html', prediction=prediction)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    email = data.get('email', '')
    if not email:
        return jsonify({'error': 'No email content provided'}), 400

    tokenized_email = cv.transform([email])
    prediction = clf.predict(tokenized_email)
    prediction_label = 'Spam' if prediction[0] == 1 else 'Non Spam'
    return jsonify({'prediction': prediction_label})

if __name__ == '__main__':
    app.run(debug=True)
