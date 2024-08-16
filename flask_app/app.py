from flask import Flask, request, jsonify
from models.knn_model import predict_knn
from models.decision_tree_model import predict_decision_tree
from models.random_forest_model import predict_random_forest

app = Flask(__name__)

# Route to predict using KNN model
@app.route('/predict_knn', methods=['POST'])
def predict_knn_route():
    try:
        data = request.get_json()
        # Extract values from the request
        checking_status = data['CheckingStatus']
        credit_history = data['CreditHistory']
        existing_savings = data['ExistingSavings']
        housing = data['Housing']
        job = data['Job']
        loan_duration = float(data['LoanDuration'])
        loan_amount = float(data['LoanAmount'])
        age = float(data['Age'])

        # Prepare the input data as expected by the model
        input_data = [
            checking_status,
            credit_history,
            existing_savings,
            housing,
            job,
            loan_duration,
            loan_amount,
            age
        ]
        
        # Make prediction
        prediction = predict_knn(input_data)
        result = 'Risk' if prediction == 1 else 'No Risk'
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route to predict using Decision Tree model
@app.route('/predict_decision_tree', methods=['POST'])
def predict_decision_tree_route():
    try:
        data = request.get_json()
        # Extract values from the request
        checking_status = data['CheckingStatus']
        credit_history = data['CreditHistory']
        existing_savings = data['ExistingSavings']
        housing = data['Housing']
        job = data['Job']
        loan_duration = float(data['LoanDuration'])
        loan_amount = float(data['LoanAmount'])
        age = float(data['Age'])

        # Prepare the input data as expected by the model
        input_data = [
            checking_status,
            credit_history,
            existing_savings,
            housing,
            job,
            loan_duration,
            loan_amount,
            age
        ]
        
        # Make prediction
        prediction = predict_decision_tree(input_data)
        result = 'Risk' if prediction == 1 else 'No Risk'
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route to predict using Random Forest model
@app.route('/predict_random_forest', methods=['POST'])
def predict_random_forest_route():
    try:
        data = request.get_json()
        # Extract values from the request
        checking_status = data['CheckingStatus']
        credit_history = data['CreditHistory']
        existing_savings = data['ExistingSavings']
        housing = data['Housing']
        job = data['Job']
        loan_duration = float(data['LoanDuration'])
        loan_amount = float(data['LoanAmount'])
        age = float(data['Age'])

        # Prepare the input data as expected by the model
        input_data = [
            checking_status,
            credit_history,
            existing_savings,
            housing,
            job,
            loan_duration,
            loan_amount,
            age
        ]
        
        # Make prediction
        prediction = predict_random_forest(input_data)
        result = 'Risk' if prediction == 1 else 'No Risk'
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
