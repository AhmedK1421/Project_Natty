from django.shortcuts import render, get_object_or_404, redirect
import json
from .models import Network, Layer, User
from .forms import NetworkForm, LayerForm
from .neural_network import Network as natty

def setup_net(request):
    """
    sets up the network by intiating x number of layers
    parameters:
        - a POST/GET request with the user
    """
    if request.method == 'POST':
        user = request.POST['user']
    elif request.method == 'GET':
        user = request.GET['user']
    context = {'user':user}
    return render(request,'neural_comps/setup.html',context)

def initiate_net(request):
    """
    parameters:
        - a POST request with the name of the network and the number of
        neurons per layer
    """
    number_of_layers = int(request.GET['number_of_layers'])
    layer_form, network_form = LayerForm(request.POST or None), NetworkForm(request.POST or None)
    if layer_form.is_valid() and network_form.is_valid():
        layer_form.save()
        layer_form = LayerForm()
        network_form.save()
        network_form = NetworkForm()
    context = {'layers': [i for i in range(1,number_of_layers+1)],
               'layer_form': layer_form,
               'network_form': network_form,
               'user': request.GET['user']
    }
    return render(request,'neural_comps/initiate_net.html',context)

def finalize_net(request):
    """
    renders a visual of the network, stores all the information about the network
    solves the network 
    parameters:
        - POST request to render visual
        - GET request to solve the network
    """
    #handeling the POST request by extracting the info and
    #using it to create a network while storing all the biases, weights, 
    #number of layers and neurons per layer
    #the whole thing is then rendered to the html file to get visual
    if request.method == 'POST':
        POST_information = {}
        for key in list(request.POST):
            POST_information[key] = []
            valuelist = request.POST.getlist(key)
            for val in valuelist:
                POST_information[key].append(val)
        username         = request.POST['username'][:len(request.POST['username'])-1]
        name             = POST_information['network_name'][0]
        child_neurons    = POST_information['number_of_neurons']
        number_of_layers = len(child_neurons)
        context = {'network':name}
        #this is the actual network
        user             = User.objects.filter(username=username)[0]
        network_object   = natty(int(child_neurons[0]),{i:int(child_neurons[i]) for i in range(1,number_of_layers)})
        biases           = network_object.layer_biases
        weights          = network_object.get_weights()
        #these are the stored network and layers models with all the information
        network = Network(network_name=name,owner=user,number_of_layers=number_of_layers)
        network.save()
        input_layer = Layer(layer_number=0,parent_network=network,number_of_neurons=int(child_neurons[0]))
        input_layer.save()
        for i in range(1,number_of_layers):
            layer = Layer(layer_number=i,parent_network=network,number_of_neurons=int(child_neurons[i]),
                          weights=weights[i-1],biases=biases[i]
            ) 
            layer.save()
    #handeling the GET request to solve the network
    #first we reconstruct the network from the stored information
    #after reconstruction we can solve the network with given values 
    #we then update the biases and weights
    #fianlly we render the neurons' to the html to display them on the visual
    #we do not update the values because we want the values to reset to zero
    elif request.method == 'GET':
        GET_information = {}
        for key in list(request.GET):
            GET_information[key] = []
            valuelist = request.GET.getlist(key)
            for val in valuelist:
                GET_information[key].append(val)
        #we use the GET request information to get the layer information
        #which we then use to extract the biases and weights
        name     = request.GET['network_name']
        username = request.GET['user']
        user = User.objects.filter(username=username)[0]
        if GET_information['inputs'][0] == '':
            return render(request,'neural_comps/setup.html',{'user':user})
        inputs   = [float(inny) for inny in GET_information['inputs']]
        network  = Network.objects.filter(network_name=name,owner=user)[0]
        layers   = Layer.objects.filter(parent_network=network)
        network_object = _get_net(layers)
        number_of_layers = len(layers)
        solution = network_object.solve_network(inputs)
        context  = {'network':name}
        ################################################################################
        """
        will be used for training when we will be updating the weights and biases
        """
        if GET_information['training'][0] != '':
            outputs   = [float(inny) for inny in GET_information['training']]
            for _ in range(1000):
                training_error = network_object.train_netowrk({tuple(inputs):outputs})
            updated_biases   = network_object.layer_biases
            updated_weights  = network_object.print_weights()
            for i in range(1,number_of_layers):
                Layer.objects.filter(parent_network=network,layer_number=i).update(biases=updated_biases[i],weights=updated_weights[i])
            context['error']= training_error
            print(training_error)
        ################################################################################
    neuron_information_extract = _get_net_info(network_object,number_of_layers)
    context['layers'] = neuron_information_extract
    context['input_layer'] = neuron_information_extract[0]
    context['user'] = username
    return render(request,'skeletal_pages/visualizing.html',context)

def to_net(request):
    """
    Parameters:
        - GET request to access the net with a name
        - Inputs might be given to solve the network
    """
    GET_information = {}
    for key in list(request.GET):
        GET_information[key] = []
        valuelist = request.GET.getlist(key)
        for val in valuelist:
            GET_information[key].append(val)
    name     = request.GET['network_name']
    user     = request.GET['user']
    owner    = User.objects.filter(username=user)[0]
    network  = Network.objects.filter(network_name=name,owner=owner)[0]
    layers   = Layer.objects.filter(parent_network=network)
    network_object = _get_net(layers)
    context  = {'network':name}
    if 'inputs' in GET_information and GET_information['inputs'] != ['']:
        inputs   = [float(inny) for inny in GET_information['inputs']]
        solution = network_object.solve_network(inputs)
    ################################################################################
        """
        will be used for training when we will be updating the weights and biases
        """
    # if 'training' in GET_information:
    if 'training' in GET_information:
        if GET_information['training'][0] != '':
                outputs   = [float(inny) for inny in GET_information['training']]
                for _ in range(1000):
                    training_error = network_object.train_netowrk({tuple(inputs):outputs})
                updated_biases   = network_object.layer_biases
                updated_weights  = network_object.print_weights()
                for i in range(1,len(layers)):
                    Layer.objects.filter(parent_network=network,layer_number=i).update(biases=updated_biases[i],weights=updated_weights[i])
                context['error'] = training_error
                print('error',training_error)
    ################################################################################
    neuron_information_extract = _get_net_info(network_object,len(layers))
    context['layers'] = neuron_information_extract
    context['input_layer'] = neuron_information_extract[0]
    context['user'] = user
    return render(request,'skeletal_pages/visualizing.html',context)

def home_page(request):
    """
    renders the home_page
    """
    return render(request,'skeletal_pages/home.html')
    
def signup(request):
    name     = request.POST['username']
    username = encrypt(name)
    password = encrypt(request.POST['password'])
    user = User(username=username,name=name,password=password)
    user.save()
    context = {'user':username}
    return render(request,'skeletal_pages/logged_in.html',context)

def login(request):
    name   = request.POST['username']
    username = encrypt(name)
    password = encrypt(request.POST['password'])
    print(request.POST)
    print(username, password)
    user   = User.objects.filter(username=username,name=name,password=password)
    if not user:
        raise TypeError("Incorrect Username or Password")
    context = {'user':username}
    return render(request,'skeletal_pages/logged_in.html',context)
    
###############################################################
#                   Helper Functions                          #
###############################################################
def _get_net(layers):
    """
    takes in the layer info and returns a network object
    """
    number_of_layers = len(layers)
    old_biases   = {}
    old_weights  = {}
    #we parse the biases and weights in the right format
    for i in range(1,number_of_layers):
        layer_bias    = layers[i].biases[1:len(layers[i].biases)-1].split(',')
        old_biases[i] = [float(bias) for bias in layer_bias]
        layer_weight  = []
        for e in range(1,len(layers[i].weights)-1):
            if layers[i].weights[e] == '[':
                starting = e
            elif layers[i].weights[e] == ']':
                parsed_weights = layers[i].weights[starting+1:e].split(',')
                if i in old_weights:
                    old_weights[i].append([float(w) for w in parsed_weights])
                    continue
                old_weights[i] = [[float(w) for w in parsed_weights]]
    biases   = {i:old_biases[i] for i in range(1,number_of_layers)}
    weights  = {i:old_weights[i] for i in range(1,number_of_layers)}
    #we reconstruct the network using the parsed information from the biases and weights
    network_object = natty(layers[0].number_of_neurons,biases,False,weights)
    return network_object

def _get_net_info(network_object,number_of_layers):
    """
    takes a network object and number of layers and returns the extracted information
    """
    neurons = [network_object.network]
    neurons += [network_object.network[0].get_layer_neurons(i) for i in range(1,number_of_layers)]
    neuron_information_extract = [[round(neurons[i][e].value,3) for e in range(len(neurons[i]))] for i in range(number_of_layers)]
    return neuron_information_extract

letters = 'abcdefghijklmnopqrstuvwxyz'
def encrypt(word):
    encryption = {letters[i]:str(i) for i in range(len(letters))}
    for num in range(10):
        encryption[str(num)] = str((num+1)*num/2)
    encrypted_word = ''
    for char in word:
        if char in encryption:
            encrypted_word += encryption[char]
            continue
        encrypted_word += char
    return encrypted_word

