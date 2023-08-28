from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# 클러스터링 모델 로드
kmeans_model = joblib.load('ctop_kmeans_model.pkl')

# 스케일러 생성
scaler = StandardScaler()

@app.route('/', methods=['GET'])
def home():
    return "hi"

@app.route('/casu', methods=['GET', 'POST'])
def cluster_data():
    try:
        if request.method == 'POST' and request.is_json:
            data = request.json  # 클라이언트에서 전송한 JSON 데이터
            new_data = np.array([[data['평균기온(°C)'], data['일교차(°C)']]])

            # 입력 데이터 표준화
            new_data_scaled = scaler.transform(new_data)

            # 클러스터 예측
            cluster_label = kmeans_model.predict(new_data_scaled)[0]

            response = {
                'cluster': int(cluster_label)
            }

            return jsonify(response), 200
        else:
            return jsonify({'error': 'Invalid request'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
