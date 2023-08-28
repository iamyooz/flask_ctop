import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)  # CORS 설정을 추가하여 CORS 관련 문제를 해결합니다.

# 클러스터링 모델 로드
kmeans_model = joblib.load('ctop_kmeans_model.pkl')

# 스케일러와 학습 데이터 로드
scaler = None
learning_data = None

def load_scaler_and_data():
    global scaler, learning_data

    # 스케일러 초기화 및 학습 데이터 로드
    if scaler is None:
        with open('ctop_learning.json', 'r') as json_file:
            learning_data = json.load(json_file)

        # 평균기온과 일교차 데이터 추출
        temperature_data = [[item['평균기온(°C)'], item['일교차(°C)']] for item in learning_data]

        # 스케일러 초기화 및 학습
        scaler = StandardScaler()
        scaler.fit(temperature_data)

@app.route('/', methods=['POST'])
def get_cluster_images():
    global scaler, learning_data

    try:
        if request.method == 'POST' and request.is_json:
            data = request.json  # 클라이언트에서 전송한 JSON 데이터

            # 스케일러 및 학습 데이터 로드
            load_scaler_and_data()

            # 입력 데이터 추출
            input_data = [[data['평균기온(°C)'], data['일교차(°C)']]]

            # 입력 데이터 표준화
            input_data_scaled = scaler.transform(input_data)

            # 클러스터 예측
            cluster_label = kmeans_model.predict(input_data_scaled)[0]

            # 클러스터와 일치하는 이미지 링크 추출
            cluster_images = []
            for item in learning_data:
                if item['cluster'] == cluster_label:
                    cluster_images.append(item['이미지'])

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
