from django.conf.urls import url
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from rest_framework.authtoken.views import obtain_auth_token
from . import views
from rest_framework_jwt.views import obtain_jwt_token

from .views import ImageView
app_name = 'user_profile'


router = DefaultRouter()
router.register(r"register2", views.UserRegisterViewsetdt)


urlpatterns = [
    path("register2/", views.RegisterView2.as_view(), name="regiser2"),
    path('register/', views.RegisterView.as_view(), name='register'),
    path('login/', obtain_jwt_token),
    path('test/', views.TestView.as_view()),
    path('images/', ImageView.as_view()),
]

