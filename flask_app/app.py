from flask import Flask, render_template, request, jsonify
from models.knn_model import predict_knn, train_knn
from models.decision_tree_model import predict_decision_tree, train_decision_tree
from models.random_forest_model import predict_random_forest, train_random_forest
from sklearn import preprocessing
import numpy as np

app = Flask(__name__)

# Train models once at the start
knn_model = train_knn()
decision_tree_model = train_decision_tree()
random_forest_model = train_random_forest()

# Encoder objects to ensure consistent encoding
label_encoder = preprocessing.LabelEncoder()
one_hot_encoder = preprocessing.OneHotEncoder()

# Assume these are the known categories for encoding (you might want to load these from your dataset)
checking_status_categories = ['no_checking', '<0', '0<=X<200', '>=200']
credit_history_categories = ['critical/other_existing_credit', 'existing_paid', 'delayed_past', 'no_credit', 'all_paid']
existing_savings_categories = ['unknown', '<100', '500<=X<1000', '>=1000', '100<=X<500']
housing_categories = ['own', 'for_free', 'rent']
job_categories = ['skilled', 'unskilled_non_resident', 'highly_skilled', 'unskilled_resident']

# Fit the encoders with categories
label_encoder.fit(checking_status_categories + existing_savings_categories)
one_hot_encoder.fit(np.array(credit_history_categories + housing_categories + job_categories).reshape(-1, 1))

@app.route('/')
def index():
    return render_template('index.html')

def encode_input_data(input_data):
    # Extract individual inputs
    checking_status = input_data[0]
    credit_history = input_data[1]
    existing_savings = input_data[2]
    housing = input_data[3]
    job = input_data[4]
    loan_duration = input_data[5]
    loan_amount = input_data[6]
    age = input_data[7]

    # Label encode for CheckingStatus and ExistingSavings
    checking_status_encoded = label_encoder.transform([checking_status])
    existing_savings_encoded = label_encoder.transform([existing_savings])

    # One-hot encode for CreditHistory, Housing, and Job
    credit_history_encoded = one_hot_encoder.transform([[credit_history]]).toarray()
    housing_encoded = one_hot_encoder.transform([[housing]]).toarray()
    job_encoded = one_hot_encoder.transform([[job]]).toarray()

    # Concatenate all the encoded features along with numerical features
    numerical_features = np.array([loan_duration, loan_amount, age]).reshape(1, -1)
    encoded_input_data = np.concatenate([checking_status_encoded.reshape(1, -1),
                                         credit_history_encoded,
                                         existing_savings_encoded.reshape(1, -1),
                                         housing_encoded,
                                         job_encoded,
                                         numerical_features], axis=1)

    return encoded_input_data

@app.route('/predict_knn', methods=['POST'])
def predict_knn_route():
    try:
        data = request.get_json()
        
        input_data = [
            data['CheckingStatus'], 
            data['CreditHistory'], 
            data['ExistingSavings'], 
            data['Housing'], 
            data['Job'], 
            data['LoanDuration'], 
            data['LoanAmount'], 
            data['Age']
        ]
        
        # Encode input data
        encoded_input_data = encode_input_data(input_data)
        
        # Get the prediction from the KNN model
        prediction = predict_knn(knn_model, encoded_input_data)
        
        # Convert prediction to Risk/No Risk
        result = "Risk" if prediction == 1 else "No Risk"
        
        return jsonify({'prediction': result})
    except Exception as e:
        print("Error during KNN prediction:", e)
        return jsonify({'error': 'An error occurred during KNN prediction'})

@app.route('/predict_decision_tree', methods=['POST'])
def predict_decision_tree_route():
    try:
        data = request.get_json()
        
        input_data = [
            data['CheckingStatus'], 
            data['CreditHistory'], 
            data['ExistingSavings'], 
            data['Housing'], 
            data['Job'], 
            data['LoanDuration'], 
            data['LoanAmount'], 
            data['Age']
        ]
        
        # Encode input data
        encoded_input_data = encode_input_data(input_data)
        
        prediction = predict_decision_tree(decision_tree_model, encoded_input_data)
        
        result = "Risk" if prediction == 1 else "No Risk"
        
        return jsonify({'prediction': result})
    except Exception as e:
        print("Error during Decision Tree prediction:", e)
        return jsonify({'error': 'An error occurred during Decision Tree prediction'})

@app.route('/predict_random_forest', methods=['POST'])
def predict_random_forest_route():
    try:
        data = request.get_json()
        
        input_data = [
            data['CheckingStatus'], 
            data['CreditHistory'], 
            data['ExistingSavings'], 
            data['Housing'], 
            data['Job'], 
            data['LoanDuration'], 
            data['LoanAmount'], 
            data['Age']
        ]
        
        # Encode input data
        encoded_input_data = encode_input_data(input_data)
        
        prediction = predict_random_forest(random_forest_model, encoded_input_data)
        
        result = "Risk" if prediction == 1 else "No Risk"
        
        return jsonify({'prediction': result})
    except Exception as e:
        print("Error during Random Forest prediction:", e)
        return jsonify({'error': 'An error occurred during Random Forest prediction'})

if __name__ == '__main__':
    app.run(debug=True)
