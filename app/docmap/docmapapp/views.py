from django.shortcuts import render
from django.http import HttpResponse

def index(request):
    return render(request, 'docmapapp/home.html')

def home(request):
    return render(request, 'docmapapp/home.html')

def wells(request):
    return render(request, 'docmapapp/wells.html')
