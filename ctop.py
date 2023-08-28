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

# 스케일러 로드
scaler = None

@app.route('/', methods=['GET', 'POST'])
def cluster_data():
    global scaler

    try:
        if request.method == 'POST' and request.is_json:
            data = request.json  # 클라이언트에서 전송한 JSON 데이터
            
            if scaler is None:
                raise ValueError("Scaler has not been fitted yet. Please fit the scaler first.")
                
            # JSON 파일에서 학습 데이터 추출
            with open('ctop_learning.json', 'r') as json_file:
                learning_data = json.load(json_file)
                
            # 평균기온과 일교차 데이터 추출
            temperature_data = [[item['평균기온(°C)'], item['일교차(°C)']] for item in learning_data]
            
            # 입력 데이터 표준화
            new_data_scaled = scaler.transform(temperature_data)

            # 클러스터 예측
            cluster_labels = kmeans_model.predict(new_data_scaled)

            response = {
                'clusters': [int(label) for label in cluster_labels]
            }

            return jsonify(response), 200
        else:
            return jsonify({'error': 'Invalid request'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # 스케일러를 여기에서 학습시켜야 합니다.
    with open('ctop_learning.json', 'r') as json_file:
        learning_data = json.load(json_file)
        
    # 평균기온과 일교차 데이터 추출
    temperature_data = [[item['평균기온(°C)'], item['일교차(°C)']] for item in learning_data]
    
    # 스케일러 학습
    scaler = StandardScaler()
    scaler.fit(temperature_data)
    
    app.run(debug=True)
