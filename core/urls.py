#first set up the urls-path

from django.urls import path
from . import views

urlpatterns = [
    path('',views.index,name='index'),
    path('result2',views.result2,name='result2'),
    path('predict',views.predict,name='predict'),
    path('record',views.record,name='record'),
    path('result',views.result,name='result'),
    path('signin',views.signin,name='signin'),
    path('upload',views.upload,name='upload'),
    path('userpage',views.userpage,name='userpage'),
    path('settings',views.settings,name='settings'),
    path('signup',views.signup,name='signup'),
    path('logout', views.logout, name='logout'),
]