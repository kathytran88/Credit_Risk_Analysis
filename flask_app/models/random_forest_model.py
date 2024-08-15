import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix

# Function to train the Random Forest model and make predictions
def train_random_forest():
    df = pd.read_csv('credit-risk-dataset.csv')
    
    # Label encoding and one-hot encoding
    one_hot_encoder = preprocessing.OneHotEncoder()
    label_encoder = preprocessing.LabelEncoder()

    CheckingStatus_encoded = label_encoder.fit_transform(df['CheckingStatus']).reshape(-1, 1)
    CreditHistory_encoded = one_hot_encoder.fit_transform(df[['CreditHistory']]).toarray()
    ExistingSavings_encoded = label_encoder.fit_transform(df['ExistingSavings']).reshape(-1, 1)
    Housing_encoded = one_hot_encoder.fit_transform(df[['Housing']]).toarray()
    Job_encoded = one_hot_encoder.fit_transform(df[['Job']]).toarray()

    # Numerical features
    numerical_features = df[['LoanDuration', 'LoanAmount', 'Age']].values
    X = np.concatenate([CheckingStatus_encoded, CreditHistory_encoded, ExistingSavings_encoded, Housing_encoded, Job_encoded, numerical_features], axis=1)

    # Target variable
    y = label_encoder.fit_transform(df[['Risk']])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

    # Train Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=1)
    rf.fit(X_train, y_train)

    # Evaluate model
    y_prediction = rf.predict(X_test)
    print(f"Accuracy on y test: {accuracy_score(y_test, y_prediction)}")

    return rf

# Function to make a prediction using the trained model
def predict_random_forest(input_data):
    rf = train_random_forest()
    prediction = rf.predict(np.array(input_data).reshape(1, -1))
    return prediction[0]
