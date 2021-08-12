from django.conf.urls import url
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

from .views import ImageView, MyJSONWebToken

app_name = 'user_profile'


router = DefaultRouter()
router.register(r"register3", views.UserRegisterViewset)

urlpatterns = [
    path("register2/", views.RegisterView2.as_view(), name="regiser2"),
    path('register/', views.RegisterView.as_view(), name='register'),
    # path('login/', obtain_jwt_token),
    path('test/', views.TestView.as_view()),
    path('images/', ImageView.as_view()),
    path('console/', ConsoleView.as_view(), "console"),
    url(r'login/$', MyJSONWebToken.as_view(), name="login")
]

urlpatterns += router.urls

