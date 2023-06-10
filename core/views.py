from django.shortcuts import render,redirect
from django.contrib.auth import authenticate,login
from django.contrib.auth.models import User, auth
from django.contrib import messages
from django.http import HttpResponse,JsonResponse
from django.contrib.auth.decorators import login_required
from .models import Profile,Uploads
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.discriminant_analysis import StandardScaler
from PIL import Image
from django.core.files.storage import default_storage
import pandas as pd
import sounddevice as sd
import librosa
import cv2
import numpy as np
import pickle
import torch


# Create your views here.
def index(request):
    return render(request,'index.html')


def userpage(request):
    try: 
        user_profile=Profile.objects.get(user=request.user)
    except Profile.DoesNotExist:
        return HttpResponse('NO PROFILE FOUND')

    user_profile = Profile.objects.get(user=request.user)
    
    return render(request,'userpage.html', {'userprofile':user_profile})

def result2(request):
    return render(request,'result2.html')


def predict(request):
    try: 
        user_profile=Uploads.objects.get(user=request.user)
    except Profile.DoesNotExist:
        return HttpResponse('NO PROFILE FOUND')
    print(user_profile)
    svm_prediction(user_profile.voice)
    #svmop = 1
    #cnnop = cnn_prediction(user_profile.image)
    #lr_prediction(svmop,cnnop)
    return render(request,'result.html')

def extract_features(signal, sr):
    features = {}
    
    # Fundamental frequency features
    pitches = librosa.yin(signal, fmin=50, fmax=300, sr=sr)
    features['MDVP:Fo(Hz)'] = pitches.mean()
    features['MDVP:Fhi(Hz)'] = pitches.max()
    features['MDVP:Flo(Hz)'] = pitches.min()
    
    # Jitter features
    features['MDVP:Jitter(%)'] = np.sqrt(np.mean(signal ** 2))
    features['MDVP:Jitter(Abs)'] = np.mean(np.abs(signal))
    features['MDVP:RAP'] = np.sqrt(np.mean(signal ** 2))
    features['MDVP:PPQ'] = np.sqrt(np.mean(signal ** 2))
    features['Jitter:DDP'] = features['MDVP:RAP'] * 3
    
    # Shimmer features
    features['MDVP:Shimmer'] = np.sqrt(np.mean(signal ** 2))
    features['MDVP:Shimmer(dB)'] = np.sqrt(np.mean(signal ** 2))
    features['MDVP:APQ'] = np.sqrt(np.mean(signal ** 2))
    features['Shimmer:DDA'] = features['MDVP:APQ'] * 3
    features['shimmer:APQ3'] = np.sqrt(np.mean(signal ** 2))
    features['shimmer:APQ5'] = np.sqrt(np.mean(signal ** 2))

    # Other features
    features['NHR'] = np.sqrt(np.mean(signal ** 2))
    features['HNR'] = np.sqrt(np.mean(signal ** 2))
    features['RPDE'] = np.sqrt(np.mean(signal ** 2))
    features['DFA'] = np.sqrt(np.mean(signal ** 2))
    features['spread1'] = np.sqrt(np.mean(signal ** 2))
    features['spread2'] = np.sqrt(np.mean(signal ** 2))
    features['D2'] = np.sqrt(np.mean(signal ** 2))
    features['PPE'] = np.sqrt(np.mean(signal ** 2))
    
    return features

def svm_prediction(input_voice):
    signal = input_voice.flatten()
    sample_rate = 22050
    extracted_features = extract_features(signal, sample_rate)
    for feature, value in extracted_features.items():
        print(f'{feature}: {value}')
    svmModel = pickle.load(open('static/assets/models/svm_model.pkl', 'rb'))

    return HttpResponse('<h1>svm prediction view</h1>')

def cnn_prediction(input_image):
    cnnModel = load_model('static/assets/models/spiral.h5')
    img = Image.open(input_image)
    img_resize=img.resize((256,256))
    img_tensor = tf.convert_to_tensor(np.array(img_resize) / 255.0)
    img_tensor = tf.expand_dims(img_tensor, 0)
    prediction = cnnModel.predict(img_tensor)
    cnn_output = prediction[0][0]
    if cnn_output < 0.6: 
        print(f'Predicted class is Healthy')
    else:
        print(f'Predicted class is Parkinson')

    return cnn_output

def lr_prediction(svm_output,cnn_output): 
    with open('static/assets/models/logistic_regression_model.pkl', 'rb') as file:
        lr_model = pickle.load(file)
    cnn_input_lr = '{:.5f}'.format(cnn_output)
    svm_input_lr = '{:.5f}'.format(svm_output)    
    new_data = [[cnn_input_lr,svm_input_lr],
        [0.00246,0],
        [0.00001,0],
        [0.00000,0],
        [0.99980,0],
        [0.99971,1],
        [1.00000,1],
        [1.00000,1]]
    scaler = StandardScaler()
    new_data_scaled = scaler.fit_transform(new_data)
    predictions = lr_model.predict(new_data_scaled)
    print("Logistic regression:")
    print(predictions[0])
        
    return HttpResponse('<h1>lr prediction view</h1>')















def upload(request):
    if request.method == 'POST':
        user=request.user.username
        global input_image
        input_image=request.FILES.get('my_image')
        input_voice=request.FILES.get('my_voice')

        new_post = Uploads.objects.create(user=user, image=input_image,voice=input_voice)
        new_post.save()
        return redirect('/predict')
    else:
        return redirect('/userpage')
    return HttpResponse('<h1>upload view</h1>')

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
            user =  User.objects.create_user(username=email, email=email,password=password)
            user.save()


             #log user in and redirect to settings page
            user_login = auth.authenticate(username=email, password=password)
            auth.login(request, user_login)


            user_model = User.objects.get(username=email)
            new_profile = Profile.objects.create(user=user_model, idusers=user_model.id,firstname=firstname,lastname=lastname,age=age,gender=gender,)
            new_profile.save()
            return redirect('settings')
        
    else:
        return render(request,'signup.html')


def record(request):
    user_profile=Uploads.objects.filter(user=request.user).order_by('created_at')
    context={
        'user_profile':user_profile,
    }

    
    return render(request,'record.html',context)

   

def result(request):
    return render(request,'result.html')
def signin(request):

    if request.method=='POST':
        username=request.POST['email']
        password=request.POST['password']

        user=auth.authenticate(request,username=username,password=password)

        if user is not None:
            login(request,user)
            return redirect('/userpage')
        else:
            messages.info(request,'credentials invalid')
            return redirect('signin')
    return render(request,'loginpage.html')



@login_required(login_url='signin')
def settings(request):
    user_profile = Profile.objects.get(user=request.user)
    if request.method == 'POST':
        firstname=request.POST['firstname']
        lastname=request.POST['lastname']
        email=request.POST['email']
        age=request.POST['age']
        gender=request.POST['gender']
        user_profile.save()
        return redirect('settings')
    return render(request,'settings.html',{'user_profile':user_profile})





@login_required(login_url='signin')    
def logout(request):
    auth.logout(request)
    return redirect('signin')

