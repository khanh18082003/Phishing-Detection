from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from quickstart.preprocessing_data import preprocessing_data, read_data
import joblib
import math
# Create your views here.
# Load mô hình của bạn
model = joblib.load("/home/khanhnguyen/Public/workspace/python_3.10/project-phishing-email-detection/DNN_model.pkl")

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
    file = request.FILES.get("file_name")
    print('file_name', file)

    body_data = read_data(file)
    print('body_data', body_data)

    preprosscessed_data = preprocessing_data(body_data)
    print(preprosscessed_data)
    
    # Xử lý và dự đoán từ mô hình
    prediction = model.predict([preprosscessed_data])
    print(prediction)
    # Trả kết quả về cho client
    return Response({"prediction": f'{round(prediction[0][0] * 100, 4)} %', "result": "Phishing" if prediction[0][0] >= 0.5 else "Legitimate"})
