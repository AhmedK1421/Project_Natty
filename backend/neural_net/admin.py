from django.contrib import admin
from .models import Network, Layer, User

# Register your models here.
admin.site.register(Network)
admin.site.register(Layer)
admin.site.register(User)