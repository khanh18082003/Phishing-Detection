from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from quickstart.preprocessing_data import preprocessing_data
import joblib
import numpy as np
# Create your views here.
# Load mô hình của bạn
model = joblib.load("/home/khanhnguyen/Public/workspace/python_3.10/thesis-phishing-email-detection/fcnn_model.pkl")

@api_view(['GET'])
def getRoutes(request):
    routes = [
        'GET api/predict',
        'POST api/predict'
    ]
    return Response({'routes': routes})

@api_view(['POST'])
def predict_view(request):
    # Nhận dữ liệu từ client
    data = request.data.get("input", "")
    preprosscessed_data = preprocessing_data(data['emailContent'])
    preprosscessed_data = np.array(preprosscessed_data).reshape(-1, 100)
    print(preprosscessed_data.shape)
    # Xử lý và dự đoán từ mô hình
    prediction = model.predict([preprosscessed_data])
    print(prediction)
    # Trả kết quả về cho client
    return Response({"prediction": "100", "emailContent": data['emailContent']})

