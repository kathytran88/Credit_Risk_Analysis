from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load models
with open('decision_tree_model.pkl', 'rb') as file:
    decision_tree_model = pickle.load(file)

with open('knn_model.pkl', 'rb') as file:
    knn_model = pickle.load(file)

with open('random_forest_model.pkl', 'rb') as file:
    random_forest_model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    if request.method == 'POST':
        # Retrieve form data
        loan_duration = float(request.form['loan_duration'])
        loan_amount = float(request.form['loan_amount'])
        installment_percent = float(request.form['installment_percent'])
        age = float(request.form['age'])
        existing_credits_count = float(request.form['existing_credits_count'])
        model_type = request.form['model_type']

        # Combine input data into a numpy array
        input_data = np.array([[loan_duration, loan_amount, installment_percent, age, existing_credits_count]])

        # Select and apply the chosen model
        if model_type == 'decision_tree':
            prediction = decision_tree_model.predict(input_data)
        elif model_type == 'knn':
            prediction = knn_model.predict(input_data)
        elif model_type == 'random_forest':
            prediction = random_forest_model.predict(input_data)

        # Convert prediction to readable text
        prediction_text = 'Risk' if prediction[0] == 1 else 'No Risk'

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)

