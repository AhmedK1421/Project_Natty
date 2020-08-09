from django.db import models
from .neural_network import Network as natty
# Create your models here.

class User(models.Model):

    username = models.CharField(max_length=100)
    name     = models.CharField(max_length=100)
    password = models.CharField(max_length=100)

    def __str__(self):
        return self.username

class Network(models.Model):

    network_name     = models.CharField(max_length=20)
    number_of_layers = models.IntegerField(default=0)
    owner            = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return 'Network: ' + self.network_name

class Layer(models.Model):

    layer_number      = models.IntegerField(default=0)
    parent_network    = models.ForeignKey(Network, on_delete=models.CASCADE)
    number_of_neurons = models.IntegerField(default=0)
    weights           = models.CharField(max_length=100000)
    biases            = models.CharField(max_length=100000)

    def __str__(self):
        return 'Layer ' + str(self.layer_number) + ' of ' + self.parent_network.network_name
