#first set up the urls-path

from django.urls import path
from . import views

urlpatterns = [
    path('',views.index,name='index'),
    path('signin',views.signin,name='signin'),
    path('userpage',views.userpage,name='userpage'),
    path('settings',views.settings,name='settings'),
    path('signup',views.signup,name='signup'),
    path('logout', views.logout, name='logout'),
]