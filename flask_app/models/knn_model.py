import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Function to train the KNN model and make predictions
def train_knn():
    df = pd.read_csv('credit-risk-dataset.csv')
    
    # One-hot encoding and label encoding
    encoder = preprocessing.LabelEncoder()
    encoder2 = preprocessing.OneHotEncoder()

    # Prepare features
    CreditHistory_encoded = encoder2.fit_transform(df[['CreditHistory']]).toarray()
    ExistingSavings_encoded = encoder.fit_transform(df[['ExistingSavings']]).reshape(-1, 1)
    CheckingStatus_encoded = encoder.fit_transform(df[['CheckingStatus']]).reshape(-1, 1)
    Housing_encoded = encoder2.fit_transform(df[['Housing']]).toarray()
    Job_encoded = encoder2.fit_transform(df[['Job']]).toarray()

    # Numerical features
    numerical_features = df[['LoanDuration', 'LoanAmount', 'Age']].values
    X = np.concatenate([CheckingStatus_encoded, CreditHistory_encoded, ExistingSavings_encoded, Housing_encoded, Job_encoded, numerical_features], axis=1)
    
    # Target variable
    y = df['Risk'].apply(lambda x: 1 if x == 'Risk' else 0).values

    # Normalize data
    X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # Train KNN model
    k = 3
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    
    # Evaluate model
    yhat = knn.predict(X_test)
    print(f'Train set accuracy: {accuracy_score(y_train, knn.predict(X_train))}')
    print(f'Test set accuracy: {accuracy_score(y_test, yhat)}')
    
    return knn

# Function to make a prediction using the trained model
def predict_knn(input_data):
    knn = train_knn()
    input_data = preprocessing.StandardScaler().fit_transform(np.array(input_data).reshape(1, -1))
    prediction = knn.predict(input_data)
    return prediction[0]
