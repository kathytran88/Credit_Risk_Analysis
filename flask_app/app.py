from flask import Flask, request, jsonify
from models.knn_model import predict_knn
from models.decision_tree_model import predict_decision_tree
from models.random_forest_model import predict_random_forest

app = Flask(__name__)

@app.route('/predict_knn', methods=['POST'])
def predict_knn_route():
    data = request.get_json()
    print("Received data for KNN:", data)
    try:
        prediction = predict_knn(data)
        return jsonify({'prediction': prediction})
    except Exception as e:
        print("Error occurred:", e)
        return jsonify({'error': str(e)})

@app.route('/predict_decision_tree', methods=['POST'])
def predict_decision_tree_route():
    data = request.get_json()
    print("Received data for Decision Tree:", data)
    try:
        prediction = predict_decision_tree(data)
        return jsonify({'prediction': prediction})
    except Exception as e:
        print("Error occurred:", e)
        return jsonify({'error': str(e)})

@app.route('/predict_random_forest', methods=['POST'])
def predict_random_forest_route():
    data = request.get_json()
    print("Received data for Random Forest:", data)
    try:
        prediction = predict_random_forest(data)
        return jsonify({'prediction': prediction})
    except Exception as e:
        print("Error occurred:", e)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
