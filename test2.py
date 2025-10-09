#!/usr/bin/python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------
# 工具函数
# ----------------------------------------
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

# ----------------------------------------
# F/G 方程项
# ----------------------------------------
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


# ----------------------------------------
# 神经网络定义
# ----------------------------------------
class BNN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.L = len(layers)-1
        self.layers = nn.ModuleList()
        for i in range(self.L):
            layer = nn.Linear(layers[i], layers[i+1])
            nn.init.normal_(layer.weight, 0, 0.05)
            nn.init.normal_(layer.bias, 0, 0.05)
            self.layers.append(layer)

    def forward_from_vector(self, vec, x):
        """通过参数向量 vec 实现前向传播，不修改模型参数"""
        pointer = 0
        y = x
        for i, layer in enumerate(self.layers):
            in_f, out_f = layer.in_features, layer.out_features
            n_w = in_f * out_f
            n_b = out_f
            w = vec[pointer:pointer + n_w].view(out_f, in_f)
            pointer += n_w
            b = vec[pointer:pointer + n_b]
            pointer += n_b
            y = F.linear(y, w, b)
            if i < self.L - 1:
                y = torch.tanh(y)
        return y

    def num_params(self):
        """返回该网络参数总数"""
        total = 0
        for layer in self.layers:
            total += layer.weight.numel() + layer.bias.numel()
        return total


# ----------------------------------------
# BPINN 模型结构
# ----------------------------------------
class BPINN:
    def __init__(self, layers_f, layers_g, l=2.0, m=0.0, s=-2.0):
        self.f_net = BNN(layers_f)
        self.g_net = BNN(layers_g)
        self.l = torch.tensor(l)
        self.m = torch.tensor(m)
        self.s = torch.tensor(s)
        self.n_f = self.f_net.num_params()
        self.n_g = self.g_net.num_params()

    def forward_from_vector(self, vec, x, u):
        """使用完整向量 vec（包含 f_net, g_net, w,A）进行前向传播"""
        n_f, n_g = self.n_f, self.n_g
        var_f = vec[:n_f]
        var_g = vec[n_f:n_f + n_g]
        w_real, w_imag, A_real, A_imag = vec[n_f + n_g:]
        f_out = self.f_net.forward_from_vector(var_f, x)
        g_out = self.g_net.forward_from_vector(var_g, u)
        f_complex = torch.view_as_complex(f_out)
        g_complex = torch.view_as_complex(g_out)
        w = torch.complex(w_real, w_imag)
        A = torch.complex(A_real, A_imag)
        return f_complex, g_complex, w, A


# ----------------------------------------
# HMC 采样器
# ----------------------------------------
class HMC:
    def __init__(self, target_log_prob_fn, init_state, step_size=0.001, num_leapfrog_steps=10, num_results=100, num_burnin=100):
        self.target_log_prob_fn = target_log_prob_fn
        self.current_state = init_state.clone().detach().requires_grad_(True)
        self.step_size = step_size
        self.num_leapfrog_steps = num_leapfrog_steps
        self.num_results = num_results
        self.num_burnin = num_burnin

    def leapfrog(self, position, momentum):
        position = position.clone().detach().requires_grad_(True)
        momentum = momentum.clone().detach()
        step_size = self.step_size

        potential = -self.target_log_prob_fn(position)
        grad_potential = torch.autograd.grad(potential, position, grad_outputs=torch.ones_like(potential))[0]
        momentum = momentum - 0.5 * step_size * grad_potential

        for i in range(self.num_leapfrog_steps):
            position = position + step_size * momentum
            position.requires_grad_(True)
            if i != self.num_leapfrog_steps - 1:
                potential = -self.target_log_prob_fn(position)
                grad_potential = torch.autograd.grad(potential, position, grad_outputs=torch.ones_like(potential))[0]
                momentum = momentum - step_size * grad_potential

        potential = -self.target_log_prob_fn(position)
        grad_potential = torch.autograd.grad(potential, position, grad_outputs=torch.ones_like(potential))[0]
        momentum = momentum - 0.5 * step_size * grad_potential
        momentum = -momentum
        return position.detach(), momentum.detach()

    def run_chain(self):
        samples = []
        losses = []
        accept_count = 0
        current_position = self.current_state.clone().detach().requires_grad_(True)

        for i in range(self.num_results + self.num_burnin):
            current_momentum = torch.randn_like(current_position)
            current_potential = -self.target_log_prob_fn(current_position)
            current_kinetic = 0.5 * torch.sum(current_momentum ** 2)
            current_hamiltonian = current_potential + current_kinetic

            proposed_position, proposed_momentum = self.leapfrog(current_position, current_momentum)
            proposed_potential = -self.target_log_prob_fn(proposed_position)
            proposed_kinetic = 0.5 * torch.sum(proposed_momentum ** 2)
            proposed_hamiltonian = proposed_potential + proposed_kinetic

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

            print(f"迭代 {i}: w_real={current_position[-4]:.4f}, w_imag={current_position[-3]:.4f}")

            # 计算误差
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
        plt.ylabel("Average % Error")
        plt.title("HMC Loss evolution")
        plt.show()

        acceptance_rate = accept_count / self.num_results
        samples = np.array(samples)
        return samples, acceptance_rate


# ----------------------------------------
# 主程序
# ----------------------------------------
if __name__ == "__main__":
    bpinn = BPINN(layers_f=[1,20,20,2], layers_g=[1,20,20,2])
    a = torch.tensor(0.4999)
    l, m, s = bpinn.l, bpinn.m, bpinn.s

    N_x, N_u = 100,100
    r_plus = (1 + torch.sqrt(1-4*a**2)) / 2
    x = torch.linspace(0,1/r_plus,N_x).view(-1,1).requires_grad_(True)
    u = torch.linspace(-1,1,N_u).view(-1,1).requires_grad_(True)

    # 初始化采样向量
    init_vec = torch.cat([
        torch.randn(bpinn.n_f + bpinn.n_g),   # 两个网络的参数
        torch.tensor([0.7, -0.1, l*(l+1)-s*(s+1), 0.0])  # w_real, w_imag, A_real, A_imag
    ])
    init_vec = init_vec.clone().detach().requires_grad_(True)

    # 定义 log posterior
    def log_post(vec):
        noise_pdef = 0.05
        noise_pdeg = 0.05
        noise_w = 0.05
        noise_A = 0.05
        prior_sigma = 1.0

        f, g, w, A = bpinn.forward_from_vector(vec, x, u)
        F0,F1,F2 = F_terms(a,w,A,s,m,x)
        G0,G1,G2 = G_terms(a,w,A,s,m,u)

        dfdt = gradients(f,x)
        d2fdt2 = gradients(dfdt,x)
        dgdt = gradients(g,u)
        d2gdt2 = gradients(dgdt,u)

        res_F = F2*d2fdt2 + F1*dfdt + F0*f
        res_G = G2*d2gdt2 + G1*dgdt + G0*g

        lik_F = dist.Normal(0., noise_pdef).log_prob(res_F.real).sum() + dist.Normal(0., noise_pdef).log_prob(res_F.imag).sum()
        lik_G = dist.Normal(0., noise_pdeg).log_prob(res_G.real).sum() + dist.Normal(0., noise_pdeg).log_prob(res_G.imag).sum()
        log_prior = dist.Normal(0, prior_sigma).log_prob(vec[:-4]).sum()
        log_prior += dist.Normal(0, noise_w).log_prob(vec[-4]).sum()
        log_prior += dist.Normal(0, noise_w).log_prob(vec[-3]).sum()
        log_prior += dist.Normal(0, noise_A).log_prob(vec[-2]).sum()
        log_prior += dist.Normal(0, noise_A).log_prob(vec[-1]).sum()

        return lik_F + lik_G + log_prior

    # 运行 HMC
    hmc = HMC(target_log_prob_fn=log_post, init_state=init_vec)
    samples, acceptance_rate = hmc.run_chain()
    print("HMC采样完成, 样本形状:", samples.shape)

    w_real_mean = np.mean(samples[-20:, -4])
    w_imag_mean = np.mean(samples[-20:, -3])

    Leaver_real = 0.85023
    Leaver_img = -0.14365
    error_real = 100*(w_real_mean - Leaver_real)/Leaver_real
    error_img = 100*(w_imag_mean - Leaver_img)/Leaver_img
    avg_err = (abs(error_real) + abs(error_img))/2
    print(f"\n最终结果:")
    print(f"w_real = {w_real_mean:.5f}, w_imag = {w_imag_mean:.5f}")
    print(f"误差: real={error_real:.3f}%, imag={error_img:.3f}%, avg={avg_err:.3f}%")