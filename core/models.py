from django.db import models
from django.contrib.auth import get_user_model

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