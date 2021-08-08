from django.urls import path
from rest_framework.authtoken.views import obtain_auth_token
from . import views
from rest_framework_jwt.views import obtain_jwt_token

app_name = 'user_profile'

urlpatterns = [
    path('register/', views.RegisterView.as_view(), name='register'),
    path('login/', obtain_jwt_token),
    path('test/', views.TestView.as_view())
]
