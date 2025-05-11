from django.urls import path
from app import views

urlpatterns = [
    path('',views.aboutpage,name = 'aboutpage'),
    path('home',views.homepage,name='homepage'),
    path('enterence',views.enterence,name = 'enterence')
]
