from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# 클러스터링 모델 로드
kmeans_model = joblib.load('ctop_kmeans_model.pkl')

@app.route('/', methods=['GET'])
def home():
    return "hi"

if __name__ == '__main__':
    app.run(debug=True)