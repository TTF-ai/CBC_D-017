from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from voiceapp import views

from django.urls import path
from . import views

from django.urls import path
from . import views

urlpatterns = [
    path('emotion-detection', views.emotion_detection, name='emotion-detection'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
