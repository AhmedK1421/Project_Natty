from rest_framework import serializers
from neural_net.models import Layer, Network

class NetworkSerializer(serializers.ModelSerializer):
    class Meta:
        model = Network
        fields = ('id', 'network_name', 'number_of_layers', 'owner')


class LayerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Layer
        fields = ('id', 'layer_number', 'parent_network', 'number_of_neurons', 'weights', 'biases')
