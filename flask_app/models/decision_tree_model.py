import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix

# Function to train the Decision Tree model and make predictions
def train_decision_tree():
    df = pd.read_csv('credit-risk-dataset.csv')
    
    # Label encoding
    encoder = preprocessing.LabelEncoder()
    CheckingStatus_encoded = encoder.fit_transform(df['CheckingStatus']).reshape(-1, 1)
    ExistingSavings_encoded = encoder.fit_transform(df['ExistingSavings']).reshape(-1, 1)

    # One hot encoding
    encoder2 = preprocessing.OneHotEncoder()
    CreditHistory_encoded = encoder2.fit_transform(df[['CreditHistory']]).toarray()
    Housing_encoded = encoder2.fit_transform(df[['Housing']]).toarray()
    Job_encoded = encoder2.fit_transform(df[['Job']]).toarray()

    # Numerical features
    numerical_features = df[['LoanDuration', 'LoanAmount', 'Age']].values
    X = np.concatenate([CheckingStatus_encoded, CreditHistory_encoded, ExistingSavings_encoded, Housing_encoded, Job_encoded, numerical_features], axis=1)

    # Target variable
    y = encoder.fit_transform(df[['Risk']])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Train Decision Tree model
    creditTree = DecisionTreeClassifier(criterion='entropy', max_depth=4)
    creditTree.fit(X_train, y_train)
    
    # Evaluate model
    y_prediction = creditTree.predict(X_test)
    print(f"Model accuracy: {accuracy_score(y_test, y_prediction)}")
    
    return creditTree

# Function to make a prediction using the trained model
def predict_decision_tree(input_data):
    creditTree = train_decision_tree()
    prediction = creditTree.predict(np.array(input_data).reshape(1, -1))
    return prediction[0]
