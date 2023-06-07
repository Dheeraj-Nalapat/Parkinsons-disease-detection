from django.db import models
from django.contrib.auth import get_user_model
import uuid
from datetime import datetime

User=get_user_model()

# Create your models here.

class Profile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    firstname=models.TextField(blank=True)
    lastname=models.TextField(blank=True)
    idusers = models.IntegerField()
    profileimg = models.ImageField(upload_to='profile_images', default='user.png')
    gender = models.TextField(blank=True)
    age= models.CharField(max_length=100, blank=True)

    def __str__(self):
        return self.user.username


class Uploads(models.Model):
    id=models.UUIDField(primary_key=True,default=uuid.uuid4)
    user=models.CharField(max_length=100)
    image=models.ImageField(upload_to='post_images')
    desc=models.TextField()
    voice=models.FileField()
    created_at=models.DateTimeField(default=datetime.now)
    result=models.IntegerField(default=0)

    def __str__(self):
        return self.user

