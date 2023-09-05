import json
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin 
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:3000", "https://web-react-wow-54ouz2lllulbggn.sel3.cloudtype.app"])

kmeans_model = joblib.load('ctop_kmeans_model.pkl')
scaler = None
learning_data = None

def load_scaler_and_data():
    global scaler, learning_data
    if scaler is None:
        with open('casual_top_cluster.json', 'r') as json_file:
            learning_data = json.load(json_file)

        temperature_data = [[item['average_temperature'], item['temperature_difference']] for item in learning_data]

        scaler = StandardScaler()
        scaler.fit(temperature_data)

@app.route('/ctop', methods=['POST'])
@cross_origin(origins=["http://localhost:3000", "https://web-react-wow-54ouz2lllulbggn.sel3.cloudtype.app"], supports_credentials=True)
def get_cluster_images():
    global scaler, learning_data

    try:
        if request.method == 'POST' and request.is_json:
            data = request.json
            load_scaler_and_data()

            input_data = [[data['average_temperature'], data['temperature_difference']]]
            input_data_scaled = scaler.transform(input_data)

            cluster_label = kmeans_model.predict(input_data_scaled)[0]

            cluster_images = []
            for item in learning_data:
                if item['cluster'] == cluster_label:
                    cluster_images.append(item['image'])

            response = {
                'cluster_images': cluster_images
            }

            return jsonify(response), 200
        else:
            return jsonify({'error': 'Invalid request'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
