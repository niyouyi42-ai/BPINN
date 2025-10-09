#!/usr/bin/python3
import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
import matplotlib.pyplot as plt


def gradients(outputs, inputs, order=1):
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
            nn.init.normal_(self.layers[-1].weight, 0, 0.05)
            nn.init.normal_(self.layers[-1].bias, 0, 0.05)

    def forward(self, x):
        y = x
        for i, layer in enumerate(self.layers):
            y = layer(y) 
            if i < self.L-1:
                y = torch.tanh(y)
        return y

class BPINN(nn.Module):
    def __init__(self, layers_f, layers_g,l=2.0, m=0.0, s=-2.0):
        super().__init__()
        self.f_net = BNN(layers_f)
        self.g_net = BNN(layers_g)

        self.l =torch.tensor(l)
        self.m =torch.tensor(m)
        self.s =torch.tensor(s)
        # 共享复数参数 w,A
        self.w_real = nn.Parameter(dist.Normal(0.7,1.0).sample())
        self.w_imag = nn.Parameter(dist.Normal(-0.1,1.0).sample())
        self.A_real = nn.Parameter(dist.Normal(l*(l+1)-s*(s+1),1).sample())
        self.A_imag = nn.Parameter(dist.Normal(0.0,1.0).sample())

    def get_parameters_vector(self):
        """获取所有参数的展平向量"""
        params = []
        for param in self.parameters():
            params.append(param.flatten())
        return torch.cat(params)

    def set_parameters_vector(self, vec):
        pointer = 0
        for param in self.parameters():
            num_elements = param.numel()
            # 重要：使用.data来修改值，保持计算图
            param.data = vec[pointer:pointer+num_elements].view_as(param).data
            pointer += num_elements

    def forward(self, x, u):
        f_out = self.f_net(x)
        g_out = self.g_net(u)
        f_complex = torch.view_as_complex(f_out)
        g_complex = torch.view_as_complex(g_out)
        w = torch.complex(self.w_real, self.w_imag)
        A = torch.complex(self.A_real, self.A_imag)
        return f_complex, g_complex, w, A

class HMC:
    def __init__(self, target_log_prob_fn, init_state, step_size=0.001, num_leapfrog_steps=10, num_results=100, num_burnin=100):
        self.target_log_prob_fn = target_log_prob_fn
        self.current_state = init_state.clone().detach().requires_grad_(True)
        self.step_size = step_size
        self.num_leapfrog_steps = num_leapfrog_steps
        self.num_results = num_results
        self.num_burnin = num_burnin

    def leapfrog(self, position, momentum):
        """leapfrog 积分"""
        position = position.clone().detach().requires_grad_(True)
        momentum = momentum.clone().detach()
        step_size = self.step_size

        # 半步更新动量
        potential = -self.target_log_prob_fn(position)
        grad_potential = torch.autograd.grad(potential, position, grad_outputs=torch.ones_like(potential))[0]
        momentum = momentum - 0.5 * step_size * grad_potential

        # 循环迭代
        for i in range(self.num_leapfrog_steps):
            # 更新位置
            position = position + step_size * momentum
            position.requires_grad_(True)

            if i != self.num_leapfrog_steps - 1:
                potential = -self.target_log_prob_fn(position)
                grad_potential = torch.autograd.grad(potential, position, grad_outputs=torch.ones_like(potential))[0]
                momentum = momentum - step_size * grad_potential

        # 最后半步更新动量
        potential = -self.target_log_prob_fn(position)
        grad_potential = torch.autograd.grad(potential, position, grad_outputs= torch.ones_like(potential))[0]
        momentum = momentum - 0.5 * step_size * grad_potential

        # 反转动量 (保证对称性)
        momentum = -momentum

        return position.detach(), momentum.detach()

    def run_chain(self):
        samples = []
        losses = []
        accept_count = 0

        current_position = self.current_state.clone().detach().requires_grad_(True)

        for i in range(self.num_results + self.num_burnin):
            current_momentum = torch.randn_like(current_position)

            # 当前哈密顿量
            current_potential = -self.target_log_prob_fn(current_position)
            current_kinetic = 0.5 * torch.sum(current_momentum ** 2)
            current_hamiltonian = current_potential + current_kinetic

            # 生成 proposal
            proposed_position, proposed_momentum = self.leapfrog(current_position, current_momentum)

            # proposal 哈密顿量
            proposed_potential = -self.target_log_prob_fn(proposed_position)
            proposed_kinetic = 0.5 * torch.sum(proposed_momentum ** 2)
            proposed_hamiltonian = proposed_potential + proposed_kinetic

            # MH 接受概率
            delta_H = (current_hamiltonian - proposed_hamiltonian).real
            acceptance_prob = torch.exp(delta_H).clamp(max=1.0)
            if torch.rand(1) < acceptance_prob:
                current_position = proposed_position.clone().detach().requires_grad_(True)
                if i >= self.num_burnin:
                    accept_count += 1
            else:
                current_position = current_position.clone().detach().requires_grad_(True)

            if i >= self.num_burnin:
                samples.append(current_position.detach().numpy())
            print(f"迭代轮数:{i},目前w实部:{current_position[-4]},目前w虚部:{current_position[-3]}")

            Leaver_real = 0.85023
            Leaver_img = -0.14365
            w_real = current_position[-4].detach().numpy()
            w_img = current_position[-3].detach().numpy()
            error_real = 100*(w_real - Leaver_real)/Leaver_real
            error_img = 100*(w_img - Leaver_img)/Leaver_img
            average_error = (np.abs(error_real) + np.abs(error_img))/2
            losses.append(average_error)

        plt.figure()
        plt.plot(losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss (-log posterior)")
        plt.title("HMC Loss evolution")
        plt.show()

        acceptance_rate = accept_count / self.num_results
        samples = np.array(samples)
        return samples, acceptance_rate


bpinn = BPINN(layers_f=[1,20,20,2], layers_g=[1,20,20,2])

a = torch.tensor(0.4999)
l = bpinn.l
m = bpinn.m
s = bpinn.s

N_x, N_u = 100,100
r_plus = (1 + torch.sqrt(1-4*a**2)) / 2
x = torch.linspace(0,1/r_plus,N_x).view(-1,1).requires_grad_(True)
u = torch.linspace(-1,1,N_u).view(-1,1).requires_grad_(True)

    # 初始变量向量展开
init_vec = bpinn.get_parameters_vector()

    # log posterior 函数
def log_post(vec):
    bpinn.set_parameters_vector(vec)

    noise_pdef = 0.05
    noise_pdeg = 0.05
    noise_w = 0.05
    noise_A = 0.05
    prior_sigma = 1.0
    
    w = torch.view_as_complex(torch.stack((bpinn.w_real,bpinn.w_imag),dim=0))
    A = torch.view_as_complex(torch.stack((bpinn.A_real,bpinn.A_imag),dim=0))
    f,g,w,A = bpinn.forward(x,u)

    F0,F1,F2 = F_terms(a,w,A,s,m,x)
    G0,G1,G2 = G_terms(a,w,A,s,m,u)

    dfdt = gradients(f,x)
    d2fdt2 = gradients(dfdt,x)
    dgdt = gradients(g,u)
    d2gdt2 = gradients(dgdt,u)

    res_F = F2*d2fdt2 + F1*dfdt + F0*f
    res_G = G2*d2gdt2 + G1*dgdt + G0*g
    lik_F_real = dist.Normal(0., noise_pdef).log_prob(res_F.real).sum()
    lik_F_imag = dist.Normal(0., noise_pdef).log_prob(res_F.imag).sum()
    lik_G_real = dist.Normal(0., noise_pdeg).log_prob(res_G.real).sum()
    lik_G_imag = dist.Normal(0., noise_pdeg).log_prob(res_G.imag).sum()
    lik_F = lik_F_real + lik_F_imag
    lik_G = lik_G_real + lik_G_imag

    log_prior = 0.

    for param in bpinn.parameters():
        if param is bpinn.w_real or param is bpinn.w_imag:
            sigma = noise_w
        elif param is bpinn.A_real or param is bpinn.A_imag:
            sigma = noise_A
        else:
            sigma = prior_sigma
        log_prior += dist.Normal(0, sigma).log_prob(param).sum()
    return lik_F + lik_G + log_prior


hmc = HMC(target_log_prob_fn=log_post, init_state=init_vec)
samples,acceptance_rate = hmc.run_chain()
print("HMC采样完成, 样本形状:", samples.shape)

w_realsum = 0
w_imagsum = 0
for i in samples:
    w_realsum += i[-4]
    w_imagsum += i[-3]
w_real = w_realsum/20
w_imag = w_imagsum/20
    
def print_results_extra(w_real,w_img):
    Leaver_real = 0.85023
    Leaver_img = -0.14365
    
    error_real = 100*(w_real - Leaver_real)/Leaver_real
        #print("Percentual error for the real frequency:\n",np.abs(error_real[:,None]))
    
    error_img = 100*(w_img - Leaver_img)/Leaver_img
        #print("Percentual error for the imaginary frequency:\n",np.abs(error_img[:,None]))
    
    
        #Print the results for each entry all each in one line, printing the real and imaginary part of w, the error for the real and imaginary part of w and the average error:
    average_error = (np.abs(error_real) + np.abs(error_img))/2
    print(f"Real part of w: {w_real:.5f}, Imaginary part of w: {w_img:.5f}, Error real: {error_real:.5f}%, Error imaginary: {error_img:.5f}%, Average error: {average_error:.5f}%")
    
print_results_extra(w_real,w_imag)