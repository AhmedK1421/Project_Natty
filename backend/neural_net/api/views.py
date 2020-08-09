from rest_framework import viewsets

from neural_net.models import Layer, Network
from .serializers import LayerSerializer, NetworkSerializer


class NetworkViewSet(viewsets.ModelViewSet):

    queryset = Network.objects.all()
    serializer_class = NetworkSerializer

class LayerViewSet(viewsets.ModelViewSet):

    queryset = Layer.objects.all()
    serializer_class = LayerSerializer