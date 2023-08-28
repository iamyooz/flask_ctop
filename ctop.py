from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# 클러스터링 모델 로드
kmeans_model = joblib.load('ctop_kmeans_model.pkl')

# 스케일러 생성
scaler = StandardScaler()

@app.route('/', methods=['GET', 'POST'])
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
    # 스케일러 피팅
    data_for_scaling = [
    {
        "일시": "2022-04-27",
        "이미지": "//image.msscdn.net/mfile_s01/_street_images/81221/280.street_img_view1626a0b0cdefe7.jpg?20220428123509",
        "평균기온(°C)": 17.7,
        "일교차(°C)": 11.3,
        "cluster": 5
    },
    {
        "일시": "2022-04-28",
        "이미지": "//image.msscdn.net/mfile_s01/_street_images/81241/280.street_img_view1626f4fdd90678.jpg?",
        "평균기온(°C)": 18.2,
        "일교차(°C)": 14.3,
        "cluster": 2
    },
    {
        "일시": "2022-04-28",
        "이미지": "//image.msscdn.net/mfile_s01/_street_images/81264/280.street_img_view1626f4c109fa69.jpg?20220428100038",
        "평균기온(°C)": 18.2,
        "일교차(°C)": 14.3,
        "cluster": 2
    }
    ]
    scaler.fit(data_for_scaling)
    app.run(debug=True)
