from django.shortcuts import render
import yaml, os, json, joblib

def index(request):
    return render(request, 'index.html')

def result(request):
    cls = joblib.load('../models/model.joblib')
    list = []
    list.append(request.GET['age'])
    list.append(request.GET['sex'])
    list.append(request.GET['bmi'])
    list.append(request.GET['children'])
    list.append(request.GET['smoker'])
    list.append(request.GET['region'])
    
    answer = cls.predict([list])
    return render(request, 'index.html', {'answer': answer[0]})
# Create your views here.
