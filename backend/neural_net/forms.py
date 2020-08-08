from django import forms
from .models import Network, Layer

class NetworkForm(forms.ModelForm):
    network_name = forms.CharField(max_length=20)
    class Meta:
        model = Network
        fields = [
            'network_name'
        ]

class LayerForm(forms.ModelForm):
    number_of_neurons = forms.IntegerField()
    class Meta:
        model = Layer
        fields = [
            'number_of_neurons'
        ]