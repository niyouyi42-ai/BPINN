#!/usr/bin/python3
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.distributions as dist
import math

def gradients(outputs, inputs, order=1):
    inputs = inputs
    outputs = outputs
    re_outputs = torch.real(outputs)
    im_outputs = torch.imag(outputs)
    if order == 1:
        d_re = torch.autograd.grad(re_outputs, inputs, grad_outputs=torch.ones_like(re_outputs), create_graph=True)[0]
        d_im = torch.autograd.grad(im_outputs, inputs, grad_outputs=torch.ones_like(im_outputs), create_graph=True)[0]
        return d_re + 1j * d_im
    elif order > 1:
        return gradients(gradients(outputs, inputs, 1), inputs, order - 1)
    else:
        return outputs

def F_terms(a,w,A,s,m,x):
    b = torch.sqrt(1 - 4*a**2)
    F0 = 2*w*(-1j + 1j*b - 2*a*m - 2j*s + 2*w - a**2*w + 2*b*w) + \
         2*(a**2*w**2 - 2*a*m*(1j + w) + 1j*(1+s)*(1j*b + 2*(1+b)*w))*x + \
         ((1+b)*(1-2j*w)*(1+s-2j*w) - 4*a**3*m*w - 2*a**4*w**2 + 2*a*(1+b)*m*(1j+2*w) + \
         2*a**2*(-2 + 1j*(5+b)*w + 2*(3+b)*w**2 + 2j*s*(1j+w)))*x**2 - 2*A*(1-x+a**2*x**2)
    F1 = (2-2*x+2*a**2*x**2)*(x*(2 - s*(-2+x) - (2+b+2j*a*m)*x + 2*a**2*x**2) + 2j*w*(-1+(1-a**2+b)*x**2))
    F2 = 2*x**2*(1-x+a**2*x**2)**2
    return F0,F1,F2

def G_terms(a,w,A,s,m,u):
    G0 = 4*(A-(m+s*u)**2 - A*u**2 + (-1+u**2)*(-s + 2*a*(1+s)*u*w - a**2*w**2)) + \
         (-1+u)**2*torch.abs(m-s)**2 + 2*(-1+u**2)*(1+2*a*(1+u)*w)*torch.abs(m+s) + \
         (1+u)**2*torch.abs(m+s)**2 + 2*(-1+u**2)*torch.abs(m-s)*(1+2*a*(-1+u)*w+torch.abs(m+s))
    G1 = -4*(u**2-1)*(2*(u+a*(-1+u**2)*w)+(-1+u)*torch.abs(m-s)+(1+u)*torch.abs(m+s))
    G2 = -4*(u**2-1)**2
    return G0,G1,G2



class BNN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.L = len(layers)-1
        self.layers = nn.ModuleList()
        for i in range(self.L):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

    def forward_from_vector(self, vec, x):
        j = 0
        y = x
        for i, layer in enumerate(self.layers):
            in_f, out_f = layer.in_features, layer.out_features
            n_w = in_f * out_f
            n_b = out_f
            w = vec[j:j + n_w].view(out_f, in_f)
            j += n_w
            b = vec[j:j + n_b]
            j += n_b
            y = F.linear(y, w, b)
            if i < self.L - 1:
                y = torch.tanh(y)
        return y

    def num_params(self):
        return sum(layer.weight.numel() + layer.bias.numel() for layer in self.layers)



class BPINN(nn.Module):
    def __init__(self, layers_f, layers_g, l=2.0, m=0.0, s=-2.0):
        super().__init__()
        self.f_net = BNN(layers_f)
        self.g_net = BNN(layers_g)
        self.l = torch.tensor(l)
        self.m = torch.tensor(m)
        self.s = torch.tensor(s)
        self.n_f = self.f_net.num_params()
        self.n_g = self.g_net.num_params()

    def forward_from_vector(self, vec, x, u):
        n_f, n_g = self.n_f, self.n_g
        var_f = vec[:n_f]
        var_g = vec[n_f:n_f + n_g]
        w_real, w_imag, A_real, A_imag = vec[n_f + n_g:]
        f_out = self.f_net.forward_from_vector(var_f, x)
        g_out = self.g_net.forward_from_vector(var_g, u)
        f_complex = torch.view_as_complex(f_out)
        g_complex = torch.view_as_complex(g_out)
        f_new = ((torch.exp(x.view(-1)-1)-1)*f_complex + 1).view(-1,1)
        g_new = ((torch.exp(u.view(-1)+1)-1)*g_complex + 1).view(-1,1)
        w = torch.complex(w_real, w_imag)
        A = torch.complex(A_real, A_imag)
        return f_new, g_new, w, A




class NUTS:
    def __init__(self, target_log_prob_fn, init_state, adapt_steps=100, num_results=100, max_tree_depth=6, target_accept=0.65, delta_max=1000.0):
        self.target_log_prob_fn = target_log_prob_fn
        self.current_state = init_state.clone().detach().requires_grad_(True)
        self.adapt_steps = adapt_steps
        self.num_results = num_results
        self.max_tree_depth = max_tree_depth
        self.target_accept = target_accept
        self.delta_max = delta_max

    def grad_log_prob(self, theta):
        # Compute gradient of posterior probability
        theta = theta.clone().detach().requires_grad_(True)
        pos = self.target_log_prob_fn(theta)
        grad = torch.autograd.grad(pos, theta, grad_outputs=torch.ones_like(pos), create_graph=False)[0]
        return grad.detach()

    def leapfrog(self, theta, r, step_size):
        grad = self.grad_log_prob(theta)
        # Half step for momentum
        r_half = r + 0.5 * step_size * grad
        # Full step for position
        theta_new = theta + step_size * r_half
        grad_new = self.grad_log_prob(theta_new)
        # Another half step for momentum
        r_new = r_half + 0.5 * step_size * grad_new
        return theta_new, r_new

    def kinetic(self, r):
        return 0.5 * torch.sum(r ** 2)

    def find_reasonable_epsilon(self, theta):
        # Initialize
        step_size = 1.0
        r = torch.randn_like(theta)
        
        # Single leapfrog step
        theta_prime, r_prime = self.leapfrog(theta, r, step_size)
        
        # Calculate probability ratio
        current_log_prob = self.target_log_prob_fn(theta) - self.kinetic(r)
        new_log_prob = self.target_log_prob_fn(theta_prime) - self.kinetic(r_prime)
        log_ratio = new_log_prob - current_log_prob
        ratio = torch.exp(log_ratio).clamp(min = 1e-8).item()
        
        # Determine direction
        a = 1 if ratio > 0.5 else -1
        
        # Iteratively adjust epsilon
        while (ratio ** a) > (2.0 ** (-a)):
            step_size = (2.0 ** a) * step_size
            theta_prime, r_prime = self.leapfrog(theta, r, step_size)
            
            new_log_prob = self.target_log_prob_fn(theta_prime) - 0.5 * torch.dot(r_prime, r_prime)
            log_ratio = new_log_prob - current_log_prob
            ratio = torch.exp(log_ratio)
        
        return step_size


    def build_tree(self, theta, r, u, v, j, step_size, theta0, r0):
        if j == 0:
            #base case - take one leapfrog step in direction v
            theta_prime, r_prime = self.leapfrog(theta, r, v * step_size)
            #check whether point is in slice
            log_prob_prime = self.target_log_prob_fn(theta_prime) - self.kinetic(r_prime)
            n_prime = 1 if u <= torch.exp(log_prob_prime) else 0
            #check energy
            s_prime = 1 if (log_prob_prime > torch.log(u) - self.delta_max) else 0

            #calculate acceptance probability
            log_prob0 = self.target_log_prob_fn(theta0) - self.kinetic(r0)
            alpha_prime = min(1.0, torch.exp(log_prob_prime - log_prob0))
            n_alpha_prime = 1

            return theta_prime, r_prime, theta_prime, r_prime, theta_prime, n_prime, s_prime, alpha_prime, n_alpha_prime
        else:
            #recursion - build the left and right subtrees  
            theta_minus, r_minus, theta_plus, r_plus, theta_prime, n_prime, s_prime, alpha_prime, n_alpha_prime = \
                self.build_tree(theta, r, u, v, j - 1, step_size, theta0, r0)
            if s_prime == 1:
                if v == -1:
                    #build the left subtree
                    theta_minus, r_minus, _, _, theta_prime2, n_prime2, s_prime2, alpha_prime2, n_alpha_prime2 = \
                        self.build_tree(theta_minus, r_minus, u, v, j - 1, step_size, theta0, r0)
                else:
                    #build the right subtree
                    _, _, theta_plus, r_plus, theta_prime2, n_prime2, s_prime2, alpha_prime2, n_alpha_prime2 = \
                        self.build_tree(theta_plus, r_plus, u, v, j - 1, step_size, theta0, r0)
                # update candidate state
                if n_prime + n_prime2 > 0 and torch.rand(()) < (n_prime2 / (n_prime + n_prime2)):
                    theta_prime = theta_prime2
                # update acceptance statistics
                alpha_prime += alpha_prime2
                n_alpha_prime += n_alpha_prime2
                #update state counts
                n_prime += n_prime2
                #check U-turn condition
                s1 = torch.dot((theta_plus - theta_minus), r_minus)
                s2 = torch.dot((theta_plus - theta_minus), r_plus)
                s_prime = s_prime2 * (s1 >= 0) * (s2 >= 0)

            return theta_minus, r_minus, theta_plus, r_plus, theta_prime, n_prime, s_prime, alpha_prime, n_alpha_prime

    def run_chain(self):
        samples = []
        theta0 = self.current_state.clone().detach().requires_grad_(True)
        step_size0 = self.find_reasonable_epsilon(theta0)
        mu = math.log(10 * step_size0)
        step_size_bar = torch.tensor(1.0)
        H_bar = 0.0
        gamma, t0, kappa = 0.05, 10.0, 0.75
        step_sizes = np.zeros(self.num_results + self.adapt_steps)

        total_accepts = 0
        total_steps = 0

        for m in range(1,self.num_results + self.adapt_steps):
            r0 = torch.rand_like(theta0)
            log_prob = self.target_log_prob_fn(theta0) - self.kinetic(r0)
            u_slice = torch.rand(()) * torch.exp(log_prob)

            theta_minus, theta_plus = theta0.clone(), theta0.clone()
            r_minus, r_plus = r0.clone(), r0.clone()
            theta_prime = theta0.clone()
            n, s, j = 1, 1, 0
            alpha_sum, n_alpha_sum = 0.0, 0.0

            while s == 1 and j < self.max_tree_depth:
                v = 1 if torch.rand(()) < 0.5 else -1
                if v == -1:
                    theta_minus, r_minus, _, _, theta_prime2, n_prime, s_prime, alpha, n_alpha = \
                        self.build_tree(theta_minus, r_minus, u_slice, v, j, step_sizes[m-1] if m>1 else step_size0, theta0, r0)
                else:
                    _, _, theta_plus, r_plus, theta_prime2, n_prime, s_prime, alpha, n_alpha = \
                        self.build_tree(theta_plus, r_plus, u_slice, v, j, step_sizes[m-1] if m>1 else step_size0, theta0, r0)
                if s_prime == 1 and torch.rand(()) < min(1.0, n_prime / n):
                    theta_prime = theta_prime2.clone()
                    if m > self.adapt_steps:
                        total_accepts += 1
                n += n_prime
                s1 = torch.dot((theta_plus - theta_minus), r_minus) 
                s2 = torch.dot((theta_plus - theta_minus), r_plus) 
                s = s_prime * (s1 >= 0) * (s2 >= 0)
                alpha_sum += alpha
                n_alpha_sum += n_alpha
                j += 1

            accept_prob = alpha_sum / max(n_alpha_sum, 1)
            if torch.rand(()) < accept_prob:
                theta0= theta_prime.clone()

            if m <= self.adapt_steps:
                H_bar = (1 - 1/(m + t0)) * H_bar + (1/(m + t0)) * (self.target_accept - accept_prob)
                log_step = torch.tensor(mu - (np.sqrt(m)/gamma) * H_bar)
                log_step_bar = (m ** (-kappa)) * log_step + (1 - m ** (-kappa)) * torch.log(step_size_bar)
                step_size_bar = torch.exp(log_step_bar)
            else:
                step_sizes[m] = step_size_bar
                samples.append(theta_prime.detach().numpy())
                total_steps += 1



            if m % 1 == 0:
                f, g, w, A = bpinn.forward_from_vector(theta_prime, x, u)
                F0, F1, F2 = F_terms(a, w, A, bpinn.s, bpinn.m, x)
                G0, G1, G2 = G_terms(a, w, A, bpinn.s, bpinn.m, u)
                dfdt = gradients(f, x); d2fdt2 = gradients(dfdt, x)
                dgdt = gradients(g, u); d2gdt2 = gradients(dgdt, u)
                res_F = F2 * d2fdt2 + F1 * dfdt + F0 * f
                res_G = G2 * d2gdt2 + G1 * dgdt + G0 * g
                err_F = torch.mean(torch.abs(res_F)).item()
                err_G = torch.mean(torch.abs(res_G)).item()
                w_real, w_imag = w.real.item(), w.imag.item()
                Leaver_real, Leaver_img = 0.85023, -0.14365
                err_real = abs(100*(w_real - Leaver_real)/Leaver_real)                    
                err_img = abs(100*(w_imag - Leaver_img)/Leaver_img)
                avg_err = 0.5 * (err_real + err_img)
                print(f"Iter {m:4d} | step_size={step_size_bar:.5e} | err_F={err_F:.2e} | err_G={err_G:.2e} | avg_err={avg_err:.2f}%")

        acceptance_rate = total_accepts / total_steps
        return np.array(samples), acceptance_rate




bpinn = BPINN(layers_f=[1,200,200,200,200,2], layers_g=[1,200,200,200,200,2])
a = torch.tensor(0.4999)
l, m, s = bpinn.l, bpinn.m, bpinn.s

N_x, N_u = 100,100
r_plus = (1 + torch.sqrt(1-4*a**2)) / 2
x = torch.linspace(0,1/r_plus,N_x).view(-1,1).requires_grad_(True)
u = torch.linspace(-1,1,N_u).view(-1,1).requires_grad_(True)

vec_params = torch.load("pinn_params.pt")
init_vec = vec_params.clone().detach().requires_grad_(True)

def log_post(vec):
    noise_pdef = torch.tensor(0.05)
    noise_pdeg = torch.tensor(0.05)
    f, g, w, A = bpinn.forward_from_vector(vec, x, u)
    F0,F1,F2 = F_terms(a,w,A,s,m,x)
    G0,G1,G2 = G_terms(a,w,A,s,m,u)
    dfdt = gradients(f,x); d2fdt2 = gradients(dfdt,x)
    dgdt = gradients(g,u); d2gdt2 = gradients(dgdt,u)
    res_F = F2*d2fdt2 + F1*dfdt + F0*f
    res_G = G2*d2gdt2 + G1*dgdt + G0*g
    lik_F = dist.Normal(0., noise_pdef).log_prob(res_F.real).sum() + dist.Normal(0., noise_pdef).log_prob(res_F.imag).sum()
    lik_G = dist.Normal(0., noise_pdeg).log_prob(res_G.real).sum() + dist.Normal(0., noise_pdeg).log_prob(res_G.imag).sum()
    return lik_F + lik_G

nuts = NUTS(target_log_prob_fn=log_post, init_state=init_vec)
samples, acceptance_rate = nuts.run_chain()
print("形状:", samples.shape, "接受率:", acceptance_rate)