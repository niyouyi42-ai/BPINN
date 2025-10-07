#!/usr/bin/python3

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt



def plot_losses(loss,a,optimiser_tuning = True,lossF=None,lossG=None):
    """
    Function that plots the evolution of the loss function(s) over the epoch.
    Receives as arguments:
    - loss - The "Global" Loss function (list of values for each epoch);
    - a - the spin parameter (int);
    - lossF - the loss of the radial solution (list of values for each epoch);
    - lossG - the loss of the angular solution (list of values for each epoch);
    """
    plt.figure()
    plt.title(f"Loss over epoch for a={a}")
    # convert y-axis to Logarithmic scale
    plt.yscale("log")
    plt.plot(loss,label="total loss")
    if None not in (lossF, lossG):
        plt.plot(lossF,label="Loss of F")
        plt.plot(lossG,label="Loss of G")
    
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    if optimiser_tuning:
        plt.savefig(f"losses_a{a}_Adam.png")
    else:
        plt.savefig(f"losses_a{a}_LBFGS.png")
    
    plt.show()


def F_terms(a,w,A,s,m,x):
    """
    All these values were calculated by Mathematica.
    Calculates The F_i terms defined in the Appendix A, each one with shape (N_x,1).
    Receives as arguments:
    - a - the spin parameter, value between 0 and 0.5 (float);
    - w - the frequency of the QNM (parameter of the Neural Network);
    - A - the separation constant of thr Teukolsky equation (parameter of the Neural Network);
    - s - spin-weight (for tensor pertubations s = -2);
    - m - Spherical harmonic indicies l and m;
    - x: vector with dimensions (N_x,1) that defines the radial space.
    """

    # Important intermediate values:
    b = np.sqrt(1 - 4*a**2) # intermediate value )

    # F0 term:
    F0 = 2*w*(-1j + 1j*b - 2*a*m - 2j*s + 2*w - a**2*w + 2*b*w) + 2*(a**2*w**2 - 2*a*m*(1j + w) + 1j*(1 + s)*(1j*b + 2*(1 + b)*w))*x + \
    ((1 + b)*(1 - 2j*w)*(1 + s - 2j*w) - 4*a**3*m*w - 2*a**4*w**2 + 2*a*(1 + b)*m*(1j + 2*w) + \
    2*a**2*(-2 + 1j*(5 + b)*w + 2*(3 + b)*w**2 + 2j*s*(1j + w)))*x**2 - 2*A*(1 - x + a**2*x**2)

    # F1 term:
    F1 =  (2 - 2*x + 2*a**2*x**2)*(x*(2 - s*(-2 + x) - (2 + b + 2j*a*m)*x + 2*a**2*x**2) + 2j*w*(-1 + (1 - a**2 + b)*x**2))

    # F2 term:
    F2 = 2*x**2*(1 - x + a**2*x**2)**2

    return F0,F1,F2 

def G_terms(a,w,A,s,m,u):
    """
    Calculates The F_i terms defined in the Appendix A, each one with shape (N_x,1).
    Receives as arguments:
    - a - the spin parameter, value between 0 and 0.5 (float);
    - w - the frequency of the QNM (parameter of the Neural Network);
    - A - the separation constant of thr Teukolsky equation (parameter of the Neural Network).
    - s - spin-weight (for tensor pertubations s = -2);
    - m - Spherical harmonic indicies l and m;
    - u: vector with dimensions (N_u,1) that defines the angular space.
    """

    G0 = 4 * (A - (m + s*u)**2 -A * u**2 + (-1 + u**2) * (-s + 2 * a * (1 + s) * u * w - a**2 * w**2)) + (-1 +u)**2 * np.abs(m - s)**2 \
    + 2 * (-1 + u**2) * (1 + 2 * a * (1 + u)*  w) * np.abs(m + s) + (1 + u)**2 * np.abs(m + s)**2 + 2 * (-1 + u**2) * np.abs(m - s) * (1 + 2*a * (-1 + u) * w + np.abs(m + s))  
    
    G1 = -4* (u**2 -1 ) * (2 * (u + a * (-1 + u**2)* w) + (-1 + u) * np.abs(m - s) + (1 + u) * np.abs(m + s))

    G2 = - 4*(u**2 - 1)**2

    return G0,G1,G2

def gradients(outputs, inputs, order = 1):
    """
    Compute the derivatives of a complex function f(x) via automatic differentiation.

    -param outputs- PyTorch complex tensor of shape (N, 1) with the values of f(x)
    -param inputs-  PyTorch real tensor of shape (N, 1) with the values of x
    -param order-   Order of the derivative (default: 1)
    -return-        PyTorch complex tensor of shape (N, 1) with the values of f'(x)
    """
   
    re_outputs = torch.real(outputs)
    im_outputs = torch.imag(outputs)
    if order == 1:
        d_re = torch.autograd.grad(re_outputs, inputs, grad_outputs=torch.ones_like(re_outputs), create_graph=True)[0]
        d_im = torch.autograd.grad(im_outputs, inputs, grad_outputs=torch.ones_like(im_outputs), create_graph=True)[0]
        return d_re + (1j) * d_im
    elif order > 1:
        return gradients(gradients(outputs, inputs, 1), inputs, order - 1)
    else:
        return outputs

class NeuralNetwork(nn.Module): 
    """
    Defines both Neural Networks, for F and G. Returns the hard enforced values of both radial and angular functions.
    Receives as arguments:
    -l,m - Spherical harmonic indicies l and m;
    - input_size_x, input_size_u - The input size of each neural network (1)
    - hidden_layers - number of hidden layers
    - neurons_per_layer - number of neurons per each hidden layer
    - ouput_size - size of the output of the neural network(=2, separation of real and imaginary part)
    - n - Spherical harmonic indice n, as default we use the fundamental mode
    - s - spin-weight (for tensor pertubations s = -2)  
    """

    def __init__(self,l,m,input_size_x = 1, input_size_u = 1, hidden_layers = 3,neurons_per_layer = 200, output_size = 2 ,n = 0,s= -2):
        super(NeuralNetwork, self).__init__()
        
        #Spherical harmonic indicies l and m and the spin weigjt s
        self.l = torch.tensor(l)
        self.m = torch.tensor(m)
        self.s = torch.tensor(s) 

        #definition of the activation of the NN
        activation = nn.Tanh()

        #Initialization of the eigenvalues of omega and A to sensible values
        self.A_real = torch.nn.Parameter(data = torch.tensor(float(l*(l+1)-s*(s+1) ) ), requires_grad = True)
        self.A_img = torch.nn.Parameter(data = torch.tensor(0.0), requires_grad = True)

        self.w_real = torch.nn.Parameter(data = torch.tensor(10.0), requires_grad = True)
        self.w_img = torch.nn.Parameter(data = torch.tensor(10.0), requires_grad = True)


        #Network for the Radial Equation (depends on x)
        self.x_network = nn.Sequential()
        self.x_network.add_module("Input",nn.Linear(input_size_x, neurons_per_layer))
        self.x_network.add_module("Input activatation",activation)
        for i in range(hidden_layers):
            self.x_network.add_module(f"Hidden layer number: {i+1} ",nn.Linear(neurons_per_layer, neurons_per_layer))
            self.x_network.add_module(f"Hidden {i+1} activation",activation)
        self.x_network.add_module("Output", nn.Linear(neurons_per_layer, output_size))
        
        #Network for the Angular Equation (depends on u)
        self.u_network = nn.Sequential()
        self.u_network.add_module("Input",nn.Linear(input_size_u, neurons_per_layer))
        self.u_network.add_module("Input activatation",activation)
        for i in range(hidden_layers):
            self.u_network.add_module(f"Hidden layer number: {i+1} ",nn.Linear(neurons_per_layer, neurons_per_layer))
            self.u_network.add_module(f"Hidden {i+1} activation",activation)
        self.u_network.add_module("Output", nn.Linear(neurons_per_layer, output_size))

        #Random initialization of the network parameters:
        """
        
        # We use the same seed to ensure that the same random numbers are used for both networks
        # Plus, we use the kaiming initialization for both networks
        # std = gain / sqrt(n_input)
        # for tanh, gain = 5/3

        std_x = 5/3 / np.sqrt(100)
        std_u = 5/3 / np.sqrt(100)
        
        """
        torch.manual_seed(42)
        
        for z in self.x_network.modules():
            if isinstance(z,nn.Linear):
                nn.init.normal_(z.weight,mean=0,std=0.05)
                nn.init.constant_(z.bias,val=0)
        
        for z in self.u_network.modules():
            if isinstance(z,nn.Linear):
                nn.init.normal_(z.weight,mean=0,std=0.01)
                nn.init.constant_(z.bias,val=0)
        



    def forward(self, x, u):
        """
        Evaluates the NN and applies the hard enforcement of normalization
        Receives as arguments:
        x,u : the vectors with dimensions (N_x,1) and (N_u,1) that defined 
            the radial and angular space, respectively. 
        """
        x_output = self.x_network(x)
        u_output = self.u_network(u)

        #Clone f and g respectively, turn two collumns into a complex number and turn into a [N_x,1] matrix
        f_complex_tensor = torch.view_as_complex(x_output)
        g_complex_tensor = torch.view_as_complex(u_output)
 

        #After joining them, one needs to hard enforce f and g:
        f_new = ((torch.exp(x.view(-1)-1)-1)*f_complex_tensor + 1).view(-1,1) #Hard Enforcement for f(x)
        g_new = ((torch.exp(u.view(-1)+1)-1)*g_complex_tensor + 1).view(-1,1) #Hard Enforcement for g(u)

    
        
        return f_new, g_new


class CustomLoss(nn.Module):
    """
    Returns the Loss Function, defined here specifically for the Teukolsky Equation
    Receives as arguments:
    - Neural Network - interpolator of the NN;
    - a - the spin parameter, value between 0 and 0.5 (float);
    - r_plus - outer horizon radii of the Kerr metric (float);
    """
    def __init__(self,NeuralNetwork,a):
        super(CustomLoss,self).__init__()

        self.NeuralNetwork = NeuralNetwork
        self.a = torch.tensor(a)
        
        self.l = NeuralNetwork.l
        self.m = NeuralNetwork.m
        self.s = NeuralNetwork.s
    
    def forward(self,x,u):

        #Recover eigenvalues
        w_real = self.NeuralNetwork.w_real
        w_img = self.NeuralNetwork.w_img
        A_real = self.NeuralNetwork.A_real
        A_img = self.NeuralNetwork.A_img

        #To call the function F_terms and G_terms, we need to convert the eigenvalues to complex numbers
        w = torch.view_as_complex(torch.stack((w_real,w_img),dim=0))
        A = torch.view_as_complex(torch.stack((A_real,A_img),dim=0))

        #Compute some commom expressions
        a = self.a
        m = self.m
        s = self.s

        # Calculate the F ang G terms for the Loss Function
        F0, F1, F2 = F_terms(a,w,A,s,m,x)
        G0, G1, G2 = G_terms(a,w,A,s,m,u)
        
        #Recover the value of the hard enforced f and g
        f,g = self.NeuralNetwork(x,u)

        # Compute the derivatives of hard enforced f and g needed for the Loss Function
        dfdt = gradients(outputs = f, inputs = x)
        d2fdt2 = gradients(outputs = dfdt, inputs = x)
        dgdt = gradients(outputs = g, inputs = u)
        d2gdt2 = gradients(outputs = dgdt, inputs = u)

        #Finally compute the Loss of f and g
        lossF = torch.mean(torch.abs(F2*d2fdt2 + F1*dfdt + F0*f))
        lossG = torch.mean(torch.abs(G2*d2gdt2 + G1*dgdt + G0*g))

        #Define the overall loss
        W = 10 #  parameterization factor that ensures that  
                # both losses are on the same order of magnitude

        loss = W*lossF + lossG

        return loss


#Comparison with Leaver's results:
def print_results(w_real_list,w_img_list):
    """
    Function that prints the results of the model and compares them with Leaver's results
    Args:
        w_real_list (list): list of real values of w
        w_img_list (list): list of imaginary values of w
    Returns:
        None
    """
    Leaver_real = np.array([0.74734, 0.75025, 0.75936 ,0.77611,0.80384,0.82401,0.84451,0.85023])
    Leaver_img = np.array([-0.17793,-0.17740,-0.17565,-0.17199,-0.16431,-0.15697,-0.14707,-0.14365])

    error_real = 100*(w_real_list - Leaver_real)/Leaver_real
    #print("Percentual error for the real frequency:\n",np.abs(error_real[:,None]))

    error_img = 100*(w_img_list - Leaver_img)/Leaver_img
    #print("Percentual error for the imaginary frequency:\n",np.abs(error_img[:,None]))


    #Print the results for each entry all each in one line, printing the real and imaginary part of w, the error for the real and imaginary part of w and the average error:
    for i in range(len(w_real_list)):
        average_error = (np.abs(error_real[i]) + np.abs(error_img[i]))/2
        print(f"Real part of w: {w_real_list[i]:.5f}, Imaginary part of w: {w_img_list[i]:.5f}, Error real: {error_real[i]:.5f}%, Error imaginary: {error_img[i]:.5f}%, Average error: {average_error:.5f}%")




# Set double precision as standard for torch
#torch.set_default_dtype(torch.float32)

# sample locations over the problem domain
N_x = 100 
N_u = 100
#Here I define x = 1 / r. However, I define it in a domain that goes from r = r_plus to r = infinity
# This means that x goes from 0 to 1/ r_plus -> Hence I will define each spacial domain for each a
u = torch.linspace(-1,1,N_u).view(-1,1).requires_grad_(True)






########## change black hole spin here
#a_k_list = [0.0,0.1,0.2,0.3,0.4,0.45,0.49,0.4999]
a_k_list = [0.4999]
















#Extract List of real and imaginary values of w
w_real_list_LBFGS = []
w_img_list_LBFGS = [] 

for a in a_k_list:

    #Define the spacial domain for each a
    r_plus = (1 + np.sqrt(1-4*a**2)) / 2
    x = torch.linspace(0,1/r_plus,N_x).view(-1,1).requires_grad_(True)
    
    # define the neural network to train
    l,m = 2,0
    model = NeuralNetwork(l,m)

    # Add parameters to L-BFGS optimizer
    optimiser_tuning = torch.optim.LBFGS(model.parameters(), lr = 0.01)
    
    #Initialize the model of the loss
    model_loss = CustomLoss(model,a)
    
    #Initialize (empty lists) that store values for plots for each a
    loss_list = []

    #Define the closure for the LBFGS optimiser
    def closure():
        optimiser_tuning.zero_grad()
        loss = model_loss(x,u)
        loss.backward()
        return loss
    
    #Train the model with the fine tuning optimiser
    for i in range(2000): 
        optimiser_tuning.step(closure)

        #record values of loss function
        loss_list.append(closure().item())
        #Print the loss every 10 iterations
        if(((i+1)%10 == 0) or (i == 0) ):
            
            print(f"LFBGS: a = {a}, loss at iteration {i+1} = {closure().item():.5f}")
            print(f"For a = {a}, omega converged to w = {model.w_real.item():.5f} {model.w_img.item():.5f}i  and to A = {model.A_real.item():.5f} + {model.A_img.item():.5f}i")

            """
            # If the two last values of the loss are the same, within a tolerance of 1e-5, we stop the training:
            if i > 0 and abs(loss_list[-1] - loss_list[-10]) < 1e-5:
                break
            
            """
    

    
    #Plot of the losses for each interesting a:

    plot_losses(loss_list, a, optimiser_tuning = False)
    print(f"For a = {a}, omega converged to w = {model.w_real.item():.5f} {model.w_img.item():.5f}i  and to A = {model.A_real.item():.5f} + {model.A_img.item():.5f}i")
    print(f" ")
    #Extract List of real and imaginary values of w
    w_real_list_LBFGS.append(model.w_real.item())
    w_img_list_LBFGS.append(model.w_img.item())




# Print the results for 250 iterations of the LBFGS optimiser:
#Comparison with Leaver's results:
def print_results_extra(w_real_list,w_img_list):
    """
    Function that prints the results of the model and compares them with Leaver's results
    Args:
        w_real_list (list): list of real values of w
        w_img_list (list): list of imaginary values of w
    Returns:
        None
    """
    Leaver_real = np.array([0.85023])
    Leaver_img = np.array([-0.14365])

    error_real = 100*(w_real_list - Leaver_real)/Leaver_real
    #print("Percentual error for the real frequency:\n",np.abs(error_real[:,None]))

    error_img = 100*(w_img_list - Leaver_img)/Leaver_img
    #print("Percentual error for the imaginary frequency:\n",np.abs(error_img[:,None]))


    #Print the results for each entry all each in one line, printing the real and imaginary part of w, the error for the real and imaginary part of w and the average error:
    for i in range(len(w_real_list)):
        average_error = (np.abs(error_real[i]) + np.abs(error_img[i]))/2
        print(f"Real part of w: {w_real_list[i]:.5f}, Imaginary part of w: {w_img_list[i]:.5f}, Error real: {error_real[i]:.5f}%, Error imaginary: {error_img[i]:.5f}%, Average error: {average_error:.5f}%")
print_results_extra(np.array(w_real_list_LBFGS),np.array(w_img_list_LBFGS))



