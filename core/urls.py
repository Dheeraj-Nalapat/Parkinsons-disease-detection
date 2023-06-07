#first set up the urls-path

from django.urls import path
from . import views

urlpatterns = [
    path('',views.index,name='index'),
    path('login',views.login,name='login'),
    path('userpage',views.userpage,name='userpage'),
    path('settings',views.settings,name='settings'),
    path('signup',views.signup,name='signup'),
]