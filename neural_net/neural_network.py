
import random
import sys

####################################
#             Neurons              #
####################################

class SourceNeuron():
    """
    creates a source neuron that takes in one value
    """
    def __init__(self,b):
        """
        we have three attributes
        1. the value of the neuron
        2. the children of the neuron
        3. the layer number to keep track of layers
        """
        self.value = 0
        self.bias = b
        self.children = {}
        self.layer = 0

    def populate_net(self,k,biases,weights=None):
        """
        takes in a list of biases and an optional list of weights
        and populates the neuron's connections based on these
        values
        """
        new_neurons = []
        for b in biases:
            new = SourceNeuron(b)
            new.layer = k
            new_neurons.append(new)
        if weights:
            self._nonrandom_populate(new_neurons,weights)
            return
        return self._random_populate(new_neurons)

    def _random_populate(self,new_neurons,c=0):
        """
        takes in a list of new neurons and adds neuron connections
        with biases given in bias and randomly assigned weights
        """
        if self.children == {}:
            for neuron in new_neurons:
                self.children[neuron] = [neuron.bias, random.randint(-10,10)/10]
            return 
        for child_neuron in self.children:
            if child_neuron not in new_neurons:
                child_neuron._random_populate(new_neurons,c+1)
        return 
        
    def _nonrandom_populate(self,new_neurons,weights,k=0):
        """
        takes in a list of new_neurons and a list of weights as well as location
        of parent neuron and adds neuron connections with given bias and weight
        """
        if self.children == {}:
            for i in range(len(new_neurons)):
                neuron = new_neurons[i]
                self.children[neuron] = [neuron.bias, weights[i][k]]
            return
        children_neurons = list(self.children.keys())
        for e in range(len(children_neurons)):
            if children_neurons[e] not in new_neurons:
                children_neurons[e]._nonrandom_populate(new_neurons, weights,e)
        return

    def __iter__(self):
        """
        returns a representation of the neuron's that can help understand
        the structure
        """
        yield (self.layer,self.children)
        for neuron in self.children:
            if neuron.children:
                yield from neuron.__iter__()
    
    def show_output(self):
        if not self.children:
            return self.value
        result = []
        result += [list(self.children.keys())[i].show_output() for i in range(len(self.children))]
        return result
    
    def get_output_neurons(self):
        if not self.children:
            return [self]
        children = list(self.children.keys())
        result = []
        result += [neuron for child in children for neuron in child.get_output_neurons()]
        return filter_dubs(result)

    def get_parent_neurons(self,child_neuron):
        if child_neuron in self.children:
            return [self]
        children = list(self.children.keys())
        result = []
        result += [neuron for child in children for neuron in child.get_parent_neurons(child_neuron)]
        return result

    def dtotalerror_dneuron(self,expected):
        children = list(self.children.keys())
        if not children[0].children:
            result = 0
            for i in range(len(children)):
                output = children[i].value
                result += 2/len(children)*(output-expected[i])*output*(1-output)*self.children[children[i]][1]
            return result
        dE_dn = 0
        for child in self.children:
            output = child.value
            dE_dn += child.dtotalerror_dneuron(expected)
        return dE_dn

    def get_layer_neurons(self,i):
        if self.layer == i-1:
            return list(self.children.keys())
        return list(self.children.keys())[0].get_layer_neurons(i)

    def len(self):
        if not self.children:
            return self.layer
        return list(self.children.keys())[0].len()

    def edit_weights(self,changes):
        children = list(self.children.keys())
        if not children[0].children:
            return
        for i in range(len(children)):
            children[i].edit(changes[self.layer][i])
            children[i].edit_weights(changes)
        return 
    
    def edit(self,changes):
        try:
            if not self.children:
                return
            children = list(self.children.keys())
            for i in range(len(children)):
                self.children[children[i]][1] -= 0.5*changes[i]
            return
        except:
            raise ImportError
    
    def get_wieghts(self,result):
        children = list(self.children.keys())
        if not children[0].children:
            return result
        layer = []
        for child in children:
            layer.append([val[1] for val in child.children.values()])
        result.append(layer)
        return children[0].get_wieghts(result)

######################################################
#               Helper Functions                     #
######################################################

def _activate_results(results, func, biases):
    """
    carrys out the activtion function on every value given
    """
    for i in range(len(results[0])):
        results[0][i] = func(results[0][i],biases[i])
    return

def weighted_sum(parent_neurons, func):
    """
    calculates the weighted value of each neuron given the previous 
    neurons and the activation function
    """
    if parent_neurons[0].children:
        values = [[neuron.value for neuron in parent_neurons]]
        weights = [[child[1] for child in neuron.children.values()] for neuron in parent_neurons]
        results = matrix_multiplier(values,weights)
        biases = [info[0] for info in parent_neurons[0].children.values()]
        _activate_results(results,func,biases)
        children = list(parent_neurons[0].children.keys())
        for i in range(len(children)):
            neuron = children[i]
            neuron.value = results[0][i]
        return weighted_sum(list(parent_neurons[0].children.keys()),func)

def transpose(mat):
    """
    returns the transposing of a matrix
    """
    result = []
    for i in range(len(mat[0])):
        new_row = []
        for elm in mat:
            new_row.append(elm[i])
        result.append(new_row)
    return result

def dot_product(vec_1,vec_2):
    """
    returns the dot product of two vectors
    """
    sumi = 0
    for i in range(len(vec_1)):
        sumi += vec_1[i]*vec_2[i]
    return sumi

def matrix_multiplier(mat_1,mat_2):
    """
    takes two matricies and returns the product
    """
    if not mat_1 and not mat_2:
        return None
    if not mat_1 or not mat_2:
        return 'Error'
    if len(mat_2) != len(mat_1[0]):
        return 'Error'
    mat_2 = transpose(mat_2)
    result = []
    for row_1 in mat_1:
        new_row = []
        for row_2 in mat_2:
            new_row.append(dot_product(row_1,row_2))
        result.append(new_row)
    return result

def get_delta_weights(delta_weights,values):
    result = []
    for weight in delta_weights:
        new_weights = []
        for val in values[0]:
            new_weights.append(weight/val)
        result.append(new_weights)
    return result

def filter_dubs(given_list):
    result = []
    for n in given_list:
        if n not in result:
            result.append(n)
    return result

def factorial(x):
    if x == 0:
        return 1
    ans = 1
    for i in range(2,x+1):
        ans *= i
    return ans

def get_primes(n):
    primes = []
    for num in range(2,n):
        if factorial(num-2)%num == 1:
            primes.append(num)
    return primes

###########################################
#            Activation Functions         #
###########################################

def sigmoid(val,bias):
    """
    one activation function
    """
    e = 2.718281828459045235360287471352662497757247093699959574966967627724076630353547
    59457138217
    val += bias
    if abs(val) < 100:
        return 1/(1+e**(-val))
    if abs(val) > 100:
        return 0
    return 1

def sigmoid_prime(val,bias):
    """
    derivative of one activation function
    """
    val = sigmoid(val,bias)
    return val*(1-val)

###########################################
#               Network                   #
###########################################

class Network():
    """
    creates a network of source neurons
    """
    def __init__(self,n,b,random_bias=True,nonrandom_weights=False):
        """
        we have two inputs:
        n: how many inputs
        b: how many layers and how many neurons per layer
        we have three attributes:
        1. the number of input neurons
        2. the biases per each layer assigned randomly
        3. the network
        """
        self.num_sources = n - 1
        self.layer_biases = b
        if random_bias:
            self.layer_biases = {layer: [random.randint(-10,10)/10 for _ in range(neuron_count)] for layer, neuron_count in b.items()}
        self.layer_weights = None
        if nonrandom_weights:
            self.layer_weights = nonrandom_weights
        self.network = self._create_network()

    def _create_network(self):
        """
        creates one origin neuron which is connected to entire network
        creates n input neurons each connected to the same network as 
        the origin
        only the first level of weights is changed
        all weights and biases are assigned randomly
        """
        origin = SourceNeuron(0)
        if not self.layer_weights:
            a = 0
            for biases in self.layer_biases.values():
                a += 1
                origin.populate_net(a,biases)
        #handling the non random case where the weights are given
        else:
            a = 0
            for layer in self.layer_biases:
                a += 1
                origin.populate_net(a,self.layer_biases[layer],self.layer_weights[layer])
        result = [origin]
        for i in range(self.num_sources):
            new = SourceNeuron(0)
            if not self.layer_weights:
                for neuron, info in origin.children.items():
                    new.children[neuron] = [info[0],random.randint(-10,10)/10]
            #handling the non random case where the weights are given
            else:
                child_neurons = list(origin.children.keys())
                for e in range(len(child_neurons)):
                    n = child_neurons[e]
                    info = origin.children[n]
                    new.children[n] = [info[0],self.layer_weights[1][e][i+1]]
            result.append(new)
        return result

    def show_solution(self,sol):
        for elm in sol:
            for value in elm:
                if isinstance(value,float) or isinstance(value,int):
                    return elm
                return self.show_solution(elm)
            
    def solve_network(self,inputs,act_func=sigmoid):
        """
        takes in a list of inputs and returns the ouputs
        """
        for i in range(len(self.network)):
            self.network[i].value = inputs[i]
        weighted_sum(self.network,act_func)
        result = self.network[0].show_output()
        if isinstance(result[0],float) or isinstance(result[0],int):
            return result
        return self.show_solution(result)
    
    def train_netowrk(self,train_data):
        outputs = [self.solve_network(test,sigmoid) for test in train_data]
        outputs = transpose(outputs)
        outputs = [sum(output) for output in outputs]
        expected = [sum([expec[i] for expec in train_data.values()]) for i in range(len(outputs))]
        length_net = self.network[0].len()
        result = []
        for i in range(1,length_net):
            neurons_of_layer = self.network[0].get_layer_neurons(i)
            result.append(self.train_hidden(neurons_of_layer,expected))
        result.append(self.train_outer(outputs,expected))
        hiddens = [transpose(elm) for elm in result[1:]]
        self.network[0].edit_weights(hiddens)
        input_changes = transpose(result[0])
        for i in range(len(self.network)):
            self.network[i].edit(input_changes[i])
        total_error = 0
        for i in range(len(outputs)):
            total_error+=(outputs[i]-expected[i])**2
        self.wieghts = self.print_wieghts()
        return total_error/len(outputs)

    def train_outer(self,outputs,expected):
        output_neurons = filter_dubs(self.network[0].get_output_neurons())
        parent_neurons = filter_dubs(self.network[0].get_parent_neurons(output_neurons[0]))
        result = []
        for i in range(len(outputs)):
            changes = []
            for parent in parent_neurons:
                dtotalerror_dweight = 2/len(outputs)*(outputs[i]-expected[i])*outputs[i]*(1-outputs[i])*parent.value
                changes.append(dtotalerror_dweight)
            result.append(changes)
        return result 

    def train_hidden(self,neurons,expected):
        result = []
        parent_neurons = filter_dubs(self.network[0].get_parent_neurons(neurons[0]))
        if neurons[0].layer == 1:
            parent_neurons = self.network
        for neuron in neurons:
            changes = []
            for parent in parent_neurons:
                dE_dn = neuron.dtotalerror_dneuron(expected)
                output = neuron.value
                dtotalerror_dweight = dE_dn*output*(1-output)*parent.value
                changes.append(dtotalerror_dweight)
            result.append(changes)
        return result

    def get_wieghts(self):
        wieghts = [[[val[1] for val in neuron.children.values()] for neuron in self.network]]
        wieghts = self.network[0].get_wieghts(wieghts)
        return [transpose(wieght) for wieght in wieghts]

    def print_wieghts(self):
        wieghts = self.get_wieghts()
        return {i+1:wieghts[i] for i in range(len(wieghts))}

######################################################
#                 Debugging Tests                    #
######################################################

def testing_functionality():
    """
    expectd results:
        test_0_result: (0.76,0.8033)
        test_1_result: 200
        test_2_result: 10
    """
    print('Testing Functionality ...')
    test_0 = Network(2,{1:[0.1,0.2],2:[0.3,0.4]},False,{1:[[0.1,0.4],[0.3,0.2]] ,2:[[0.5,0.7],[0.6,0.8]]})
    test_0_result = test_0.solve_network([1,2,3],sigmoid)
    if round(test_0_result[0],2) != 0.76 or round(test_0_result[1],2) != 0.80:
        return 'test 0 fails'
    print('Test 0 passes')
    test_1 = Network(100,{1:2,2:10,3:15,4:10,5:10,6:200})
    test_1_result = test_1.solve_network([random.randint(-10,10)/10 for _ in range(100)],sigmoid)
    if len(test_1_result) != 200:
        return 'Test 1 fails'
    print('Test 1 passes')
    test_2 = Network(784,{1:16,2:16,3:10})
    test_2_result = test_2.solve_network([random.randrange(0,256) for _ in range(784)],sigmoid)
    if len(test_2_result) != 10:
        return 'Test 2 fails'
    print('Test 2 passes')
    return 'OK'

def testing_training():
    """
    expected results:
        error gets smaller
    """
    print('Testing Training ...')
    test_0 = Network(2,{1:[0.35,0.35],2:[0.6,0.6]},False,{1:[[0.15,0.20],[0.25,0.3]],2:[[0.4,0.45],[0.5,.55]]})
    print(test_0.layer_weights)
    test_0_error_1 = test_0.train_netowrk({(0.05,.10):[0.01,.99]})
    for _ in range(10000-1):
        test_0_error_2 = test_0.train_netowrk({(0.05,.10):[0.01,.99]})
    print(test_0.print_wieghts(),test_0.solve_network([0.05,0.10]))
    if test_0_error_1 < test_0_error_2:
        return 'test 0 fails\nerror 1: '+str(test_0_error_1)+'\nerror 2: '+str(test_0_error_2)
    print('Test 0 passes')
    test_1 = Network(3,{1:2,2:4,3:4})
    test_1_error_1 = test_1.train_netowrk({(0,0,1):[0,1,0,0]})
    for _ in range(10000-1):
        test_1_error_2 = test_1.train_netowrk({(0,0,1):[0,1,0,0]})
    if test_1_error_1 < test_1_error_2:
        return 'test 1 fails\nerror 1: '+str(test_1_error_1)+'\nerror 2: '+str(test_1_error_2)
    print('Test 1 passes')
    return 'OK'

###################################################
#                      REPOS                      #
###################################################

def Prime_REPO():
    layers = input('How many layers:')
    b = {}
    for layer in range(1,int(layers)-1):
        b[layer] = int(input('How many neurons in layer '+str(layer)+' :'))
    b[len(layers)-1] = 2
    prime_net = Network(1,b)
    primes = get_primes(100)
    data = {}
    for num in range(max(primes)):
        if num in primes:
            data[(num/10,)] = [0.01,0.1]
            continue
        data[(num/10,)] = [0.1,0.01]
    for key in data:
        for _ in range(10000):
            prime_error = prime_net.train_netowrk({key:data[key]})
    print('Training Complete ...')
    return _Prime_REPO_Helper(prime_net,prime_error)

def _Prime_REPO_Helper(prime_net,prime_error):
    input_value = input('Enter Number: ')
    if input_value == 'STOP':
        return 'We Done'
    solution = prime_net.solve_network([int(input_value)])
    print(solution)
    return _Prime_REPO_Helper(prime_net,prime_error)

if __name__ == '__main__':
    print(testing_functionality())
    print(testing_training())
    