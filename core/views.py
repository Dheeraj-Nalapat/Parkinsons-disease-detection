from django.shortcuts import render,redirect
from django.contrib.auth.models import User, auth
from django.contrib import messages
from django.http import HttpResponse
from .models import Profile

# Create your views here.
def index(request):
    return render(request,'index.html')

def login(request):
    return render(request,'loginpage.html')


def signup(request):
    if request.method == 'POST':
        firstname = request.POST['firstname']
        lastname = request.POST['lastname']
        email = request.POST['email']
        age =  request.POST['age']
        gender = request.POST['gender']
        password = request.POST['password']

        if User.objects.filter(email=email).exists():
            messages.info(request,'Email taken')
            return redirect('signup')
        else:
            user =  User.objects.create_user(username=firstname, email=email,password=password)
            user.save()


            user_model = User.objects.get(username=firstname)
            new_profile = Profile.objects.create(user=user_model, idusers=user_model.id)
            new_profile.save()
            return redirect('signup')
        
    else:
        return render(request,'signup.html')