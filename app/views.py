from django.shortcuts import render

# Create your views here.

def aboutpage(request):
    return render(request, 'about.html')

def homepage(request):
    return render(request, 'homepage.html')

def enterence(request):
    return(render(request, 'enterence.html'))