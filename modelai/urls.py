from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('workout', views.index, name='index'),
    path('api/frame/', views.get_frame, name='get_frame'),
]
