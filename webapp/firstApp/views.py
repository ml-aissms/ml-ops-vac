from django.shortcuts import render
import yaml, os, json, joblib
import psycopg2
from .models import mlops

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

    b = mlops(age = int(request.GET['age']),sex = int(request.GET['sex']),bmi = float(request.GET['bmi']),children = int(request.GET['children']),smoker = int(request.GET['smoker']),region = int(request.GET['region']),charges = float(answer[0]))
    b.save()
    return render(request, 'index.html', {'answer': answer[0]})
# Create your views here.
