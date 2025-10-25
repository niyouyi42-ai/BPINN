#!/usr/bin/python3
import torch
import torch.nn as nn
import numpy as np
import torch.distributions as dist
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ========== 1️⃣ 工具函数：导数、方程项 ==========
def gradients(outputs, inputs, order=1):
    re_outputs = torch.real(outputs)
    im_outputs = torch.imag(outputs)
    if order == 1:
        d_re = torch.autograd.grad(re_outputs, inputs, grad_outputs=torch.ones_like(re_outputs),
                                   create_graph=True)[0]
        d_im = torch.autograd.grad(im_outputs, inputs, grad_outputs=torch.ones_like(im_outputs),
                                   create_graph=True)[0]
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


# ========== 2️⃣ BPINN 定义（冻结网络权重） ==========
class BNN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.L = len(layers)-1
        self.layers = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(self.L)])
    def forward_from_vector(self, vec, x):
        j = 0; y = x
        for i, layer in enumerate(self.layers):
            in_f, out_f = layer.in_features, layer.out_features
            n_w = in_f * out_f; n_b = out_f
            w = vec[j:j + n_w].view(out_f, in_f); j += n_w
            b = vec[j:j + n_b]; j += n_b
            y = torch.tanh(F.linear(y, w, b)) if i < self.L - 1 else F.linear(y, w, b)
        return y
    def num_params(self):
        return sum(layer.weight.numel() + layer.bias.numel() for layer in self.layers)

class BPINN(nn.Module):
    def __init__(self, layers_f, layers_g, l=2.0, m=0.0, s=-2.0):
        super().__init__()
        self.f_net = BNN(layers_f)
        self.g_net = BNN(layers_g)
        self.l = torch.tensor(l); self.m = torch.tensor(m); self.s = torch.tensor(s)
        self.n_f = self.f_net.num_params(); self.n_g = self.g_net.num_params()

    def freeze_networks(self):
        for p in self.f_net.parameters(): p.requires_grad = False
        for p in self.g_net.parameters(): p.requires_grad = False




bpinn = BPINN(layers_f=[1,200,200,200,200,2], layers_g=[1,200,200,200,200,2])
bpinn.freeze_networks()

a = torch.tensor(0.4999)
x = torch.linspace(0,1/( (1+torch.sqrt(1-4*a**2))/2 ),100).view(-1,1).requires_grad_(True)
u = torch.linspace(-1,1,100).view(-1,1).requires_grad_(True)
l ,m, s = bpinn.l, bpinn.m, bpinn.s

# 加载训练好的参数（已固定）
vec_params = torch.load("pinn_params.pt")
w_real = vec_params[-4].item()
w_imag = vec_params[-3].item()
A_real = vec_params[-2].item()
A_imag = vec_params[-1].item()
init_vec = torch.tensor([w_real, w_imag, A_real, A_imag], dtype=torch.float32)
scales = torch.tensor([1.0, 1.0, 10.0, 10.0])  # 按量级调整
scaled_init = init_vec / scales



f_out = bpinn.f_net.forward_from_vector(vec_params[:bpinn.n_f], x)
g_out = bpinn.g_net.forward_from_vector(vec_params[bpinn.n_f:bpinn.n_f+bpinn.n_g], u)
f_complex = torch.view_as_complex(f_out)
g_complex = torch.view_as_complex(g_out)
dfdt = gradients(f_complex, x); d2fdt2 = gradients(dfdt, x)
dgdt = gradients(g_complex, u); d2gdt2 = gradients(dgdt, u)

# ========== 4️⃣ 定义 4 维后验 log_prob 函数 ==========
def log_post_4D(vec):
    wr, wi, Ar, Ai = vec*scales

    w = torch.complex(wr, wi)
    A = torch.complex(Ar, Ai)

    F0,F1,F2 = F_terms(a, w, A, s, m, x)
    G0,G1,G2 = G_terms(a, w, A, s, m, u)
    res_F = F2*d2fdt2 + F1*dfdt + F0*f_complex
    res_G = G2*d2gdt2 + G1*dgdt + G0*g_complex
    noise = 1.0
    lik_F = dist.Normal(0., noise).log_prob(res_F.real).sum() + dist.Normal(0., noise).log_prob(res_F.imag).sum()
    lik_G = dist.Normal(0., noise).log_prob(res_G.real).sum() + dist.Normal(0., noise).log_prob(res_G.imag).sum()
    return lik_F + lik_G


# ========== 5️⃣ NUTS 实现（只对4维向量） ==========
class NUTS:
    def __init__(self, target_log_prob_fn, init_state, step_size=0.0004, adapt_steps=500, num_results=1500,
                 max_tree_depth=5, target_accept=0.65):
        self.target_log_prob_fn = target_log_prob_fn
        self.current_state = init_state.clone().detach().requires_grad_(True)
        self.step_size = step_size; self.adapt_steps = adapt_steps
        self.num_results = num_results; self.max_tree_depth = max_tree_depth
        self.target_accept = target_accept

    def grad_log_prob(self, theta):
        theta = theta.clone().detach().requires_grad_(True)
        logp = self.target_log_prob_fn(theta)
        grad = torch.autograd.grad(logp, theta, torch.ones_like(logp))[0]
        return logp.detach(), grad.detach()

    def leapfrog(self, theta, r, grad, step_size):
        r_half = r + 0.5 * step_size * grad
        theta_new = theta + step_size * r_half
        logp_new, grad_new = self.grad_log_prob(theta_new)
        r_new = r_half + 0.5 * step_size * grad_new
        return theta_new, r_new, grad_new, logp_new

    def kinetic(self, r): return 0.5 * torch.sum(r ** 2)

    def build_tree(self, theta, r, grad, log_u, v, j, step_size, logp0, delta_max=100.0):
        if j == 0:
            theta_new, r_new, grad_new, logp_new = self.leapfrog(theta, r, grad, v * step_size)
            joint = logp_new - self.kinetic(r_new)
            n = int(log_u <= joint)
            s = int(log_u < delta_max + joint)
            alpha = min(1.0, math.exp(joint - (logp0 - self.kinetic(r))))
            return theta_new, r_new, grad_new, theta_new, r_new, grad_new, theta_new, n, s, alpha, 1
        else:
            t_minus, r_minus, g_minus, t_plus, r_plus, g_plus, theta_prime, n_prime, s_prime, alpha, n_alpha = \
                self.build_tree(theta, r, grad, log_u, v, j - 1, step_size, logp0)
            if s_prime == 1:
                if v == -1:
                    t_minus, r_minus, g_minus, _, _, _, theta_prime2, n2, s2, alpha2, n_a2 = \
                        self.build_tree(t_minus, r_minus, g_minus, log_u, v, j - 1, step_size, logp0)
                else:
                    _, _, _, t_plus, r_plus, g_plus, theta_prime2, n2, s2, alpha2, n_a2 = \
                        self.build_tree(t_plus, r_plus, g_plus, log_u, v, j - 1, step_size, logp0)
                if n_prime + n2 > 0 and torch.rand(()) < n2 / (n_prime + n2):
                    theta_prime = theta_prime2
                n_prime += n2; s_prime = int(s_prime and s2)
                alpha += alpha2; n_alpha += n_a2
            return t_minus, r_minus, g_minus, t_plus, r_plus, g_plus, theta_prime, n_prime, s_prime, alpha, n_alpha

    def run_chain(self):
        samples = []; current_theta = self.current_state.clone().detach().requires_grad_(True)
        logp, grad = self.grad_log_prob(current_theta)
        step_size = self.step_size; mu = math.log(10 * step_size)
        h_bar, log_eps_bar = 0, 0; gamma, t0, kappa = 0.05, 10.0, 0.75
        accept_sum, total_count = 0, 0

        for t in range(1, self.num_results + self.adapt_steps + 1):
            r0 = torch.randn_like(current_theta)
            log_u = (logp - self.kinetic(r0)).item() - np.random.exponential(1.0)
            t_minus, t_plus = current_theta.clone(), current_theta.clone()
            r_minus, r_plus = r0.clone(), r0.clone()
            g_minus, g_plus = grad.clone(), grad.clone()
            theta_prime = current_theta.clone()
            n, s, j = 1, 1, 0; alpha_sum, n_alpha = 0.0, 0.0

            while s == 1 and j < self.max_tree_depth:
                v = 1 if torch.rand(()) < 0.5 else -1
                if v == -1:
                    t_minus, r_minus, g_minus, _, _, _, theta_prime2, n2, s2, a2, n_a2 = \
                        self.build_tree(t_minus, r_minus, g_minus, log_u, v, j, step_size, logp)
                else:
                    _, _, _, t_plus, r_plus, g_plus, theta_prime2, n2, s2, a2, n_a2 = \
                        self.build_tree(t_plus, r_plus, g_plus, log_u, v, j, step_size, logp)
                if s2 == 1 and torch.rand(()) < n2 / max(n + n2, 1):
                    theta_prime = theta_prime2.clone()
                n += n2; s = s2; alpha_sum += a2; n_alpha += n_a2; j += 1

            accept_prob = alpha_sum / max(n_alpha, 1)
            if torch.rand(()) < accept_prob:
                current_theta = theta_prime.clone()
                logp, grad = self.grad_log_prob(current_theta)

            # Step size adapt
            if t < self.adapt_steps:
                h_bar = (1 - 1/(t + t0)) * h_bar + (1/(t + t0)) * (self.target_accept - accept_prob)
                log_step = mu - (math.sqrt(t)/gamma) * h_bar
                step_size = math.exp(log_step)
                log_eps_bar = (t**(-kappa)) * log_step + (1 - t**(-kappa)) * log_eps_bar
            else:
                step_size = math.exp(log_eps_bar)
                samples.append(current_theta.detach().cpu().numpy())
                accept_sum += accept_prob; total_count += 1

            if t % 10 == 0:
                print(f"Iter {t:4d}, step_size={step_size:.5e}, accept={accept_prob:.3f}")

        acc_rate = accept_sum / max(total_count, 1)
        return np.array(samples), acc_rate


# ========== 6️⃣ 运行采样 ==========
nuts = NUTS(target_log_prob_fn=log_post_4D, init_state=scaled_init)
samples, acc_rate = nuts.run_chain()
print("接受率:", acc_rate)

# ========== 7️⃣ 计算 ESS ==========
def autocorr(x, lag):
    x = np.array(x) - np.mean(x)
    return np.sum(x[:-lag]*x[lag:]) / np.sum(x*x) if lag < len(x) else 0
def effective_sample_size(x):
    n = len(x); acf_sum = 0
    for lag in range(1,n):
        rho = autocorr(x, lag)
        if rho <= 0: break
        acf_sum += 2*rho
    return n / (1 + acf_sum)

samples = np.array(samples)
names = ["w_real","w_imag","A_real","A_imag"]
for i,name in enumerate(names):
    ess = effective_sample_size(samples[:,i])
    print(f"ESS({name}) = {ess:.1f}")

# ========== 8️⃣ 绘制采样轨迹 ==========
plt.figure(figsize=(10,6))
for i,name in enumerate(names):
    plt.subplot(4,1,i+1)
    plt.plot(samples[:,i])
    plt.ylabel(name)
plt.tight_layout()
plt.show()