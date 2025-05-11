from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('app.urls')),
    path('', include('authentication.urls')),
    path('', include('modelai.urls')),
    path('', include('voiceapp.urls')),
    path('', include('modelai.urls')),
  # Point to your app
]
