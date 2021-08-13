"""illegal_fund_raising_forecast URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.shortcuts import render
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings

def index(request):
    return render(request, 'index.html')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('server/api/user/', include("user_profile.urls", namespace='profile')),
    path('server/api/', include("forecast.urls", namespace='forecast')),
    path('server/api/index/', index)
]

urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

# 配置图像的url
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

