#!/usr/bin/python3
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.distributions as dist
import math

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
    def __init__(self, target_log_prob_fn, init_state, step_size=0.01, adapt_steps=1500, num_results=1500, max_tree_depth=10, target_accept=0.65):
        self.target_log_prob_fn = target_log_prob_fn
        self.current_state = init_state.clone().detach().requires_grad_(True)
        self.step_size = step_size
        self.adapt_steps = adapt_steps
        self.num_results = num_results
        self.max_tree_depth = max_tree_depth
        self.target_accept = target_accept

    def grad_log_prob(self, theta):
        theta = theta.clone().detach().requires_grad_(True)
        logp = self.target_log_prob_fn(theta)
        grad = torch.autograd.grad(logp, theta, grad_outputs=torch.ones_like(logp), create_graph=False)[0]
        return logp.detach(), grad.detach()

    def leapfrog(self, theta, r, grad, step_size):
        r_half = r + 0.5 * step_size * grad
        theta_new = theta + step_size * r_half
        logp_new, grad_new = self.grad_log_prob(theta_new)
        r_new = r_half + 0.5 * step_size * grad_new
        return theta_new, r_new, grad_new, logp_new

    def kinetic(self, r):
        return 0.5 * torch.sum(r ** 2)

    def build_tree(self, theta, r, grad, log_u, v, j, step_size, logp0, delta_max=1000.0):
        if j == 0:
            theta_new, r_new, grad_new, logp_new = self.leapfrog(theta, r, grad, v * step_size)
            joint = logp_new - self.kinetic(r_new)
            n = int(log_u <= joint)
            s = int(log_u < delta_max + joint)
            alpha = min(1.0, math.exp(joint - (logp0 - self.kinetic(r))))
            n_alpha = 1
            return theta_new, r_new, grad_new, theta_new, r_new, grad_new, theta_new, n, s, alpha, n_alpha
        else:
            thetaminus, rminus, gradminus, thetaplus, rplus, gradplus, thetaprime, nprime, sprime, alpha, n_alpha = \
                self.build_tree(theta, r, grad, log_u, v, j - 1, step_size, logp0)
            if sprime == 1:
                if v == -1:
                    thetaminus, rminus, gradminus, _, _, _, thetaprime2, nprime2, sprime2, alpha2, n_alpha2 = \
                        self.build_tree(thetaminus, rminus, gradminus, log_u, v, j - 1, step_size, logp0)
                else:
                    _, _, _, thetaplus, rplus, gradplus, thetaprime2, nprime2, sprime2, alpha2, n_alpha2 = \
                        self.build_tree(thetaplus, rplus, gradplus, log_u, v, j - 1, step_size, logp0)
                if nprime + nprime2 > 0 and torch.rand(()) < nprime2 / (nprime + nprime2):
                    thetaprime = thetaprime2
                nprime += nprime2
                s1 = ((thetaplus - thetaminus) @ rminus) >= 0
                s2 = ((thetaplus - thetaminus) @ rplus) >= 0
                sprime = int(sprime and sprime2 and s1 and s2)
                alpha += alpha2
                n_alpha += n_alpha2
            return thetaminus, rminus, gradminus, thetaplus, rplus, gradplus, thetaprime, nprime, sprime, alpha, n_alpha

    def run_chain(self):
        samples = []
        current_theta = self.current_state.clone().detach().requires_grad_(True)
        logp, grad = self.grad_log_prob(current_theta)
        step_size = self.step_size
        mu = math.log(10 * step_size)
        h_bar, log_eps_bar = 0, 0
        gamma, t0, kappa = 0.05, 10.0, 0.75
        accept_sum, total_count = 0, 0

        for t in range(1,self.num_results + self.adapt_steps):
            m = t - 1
            r0 = torch.randn_like(current_theta)
            log_u = (logp - self.kinetic(r0)).item() - np.random.exponential(1.0)

            thetaminus, thetaplus = current_theta.clone(), current_theta.clone()
            rminus, rplus = r0.clone(), r0.clone()
            gradminus, gradplus = grad.clone(), grad.clone()
            theta_prime = current_theta.clone()
            n, s, j = 1, 1, 0
            alpha_sum, n_alpha = 0.0, 0.0

            while s == 1 and j < self.max_tree_depth:
                v = 1 if torch.rand(()) < 0.5 else -1
                if v == -1:
                    thetaminus, rminus, gradminus, _, _, _, theta_prime2, nprime, sprime, alpha, n_alpha_ = \
                        self.build_tree(thetaminus, rminus, gradminus, log_u, v, j, step_size, logp)
                else:
                    _, _, _, thetaplus, rplus, gradplus, theta_prime2, nprime, sprime, alpha, n_alpha_ = \
                        self.build_tree(thetaplus, rplus, gradplus, log_u, v, j, step_size, logp)
                if sprime == 1 and torch.rand(()) < nprime / max(n + nprime, 1):
                    theta_prime = theta_prime2.clone()
                n += nprime
                s = sprime and ((thetaplus - thetaminus) @ rminus >= 0) and ((thetaplus - thetaminus) @ rplus >= 0)
                alpha_sum += alpha
                n_alpha += n_alpha_
                j += 1

            accept_prob = alpha_sum / max(n_alpha, 1)
            if torch.rand(()) < accept_prob:
                current_theta = theta_prime.clone()
                logp, grad = self.grad_log_prob(current_theta)

            if t < self.adapt_steps:
                h_bar = (1 - 1/(t + t0)) * h_bar + (1/(t + t0)) * (self.target_accept - accept_prob)
                log_step = mu - (math.sqrt(t)/gamma) * h_bar
                step_size = math.exp(log_step)
                log_eps_bar = (t ** (-kappa)) * log_step + (1 - t ** (-kappa)) * log_eps_bar
            else:
                step_size = math.exp(log_eps_bar)
                samples.append(current_theta.detach().numpy())
                accept_sum += accept_prob
                total_count += 1

            if m % 10 == 0:
                f, g, w, A = bpinn.forward_from_vector(current_theta, x, u)
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
                print(f"Iter {t:4d} | step_size={step_size:.5e} | accept={accept_prob:.3f} | err_F={err_F:.2e} | err_G={err_G:.2e} | avg_err={avg_err:.2f}%")

        acceptance_rate = accept_sum / max(total_count, 1)
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
    noise_pdef = 0.05
    noise_pdeg = 0.05
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




w_realsum = 0
w_imagsum = 0
for i in samples:
    w_realsum += i[-4]
    w_imagsum += i[-3]
w_real = w_realsum/1500
w_imag = w_imagsum/1500

def print_results_extra(w_real,w_img):
    Leaver_real = 0.85023
    Leaver_img = -0.14365

    error_real = np.abs(100*(w_real - Leaver_real)/Leaver_real)

    error_img = np.abs(100*(w_img - Leaver_img)/Leaver_img)

    average_error = (np.abs(error_real) + np.abs(error_img))/2
    print(f"Real part of w: {w_real:.5f}, Imaginary part of w: {w_img:.5f}, Error real: {error_real:.5f}%, Error imaginary: {error_img:.5f}%, Average error: {average_error:.5f}%")

print_results_extra(w_real,w_imag)





vec_last = torch.tensor(samples[-1], dtype=torch.float32, requires_grad=True)


f, g, w, A = bpinn.forward_from_vector(vec_last, x, u)
F0, F1, F2 = F_terms(a, w, A, s, m, x)
G0, G1, G2 = G_terms(a, w, A, s, m, u)

dfdt = gradients(f, x)
d2fdt2 = gradients(dfdt, x)
dgdt = gradients(g, u)
d2gdt2 = gradients(dgdt, u)

res_F = F2 * d2fdt2 + F1 * dfdt + F0 * f
res_G = G2 * d2gdt2 + G1 * dgdt + G0 * g


error_F = torch.mean(torch.abs(res_F))
error_G = torch.mean(torch.abs(res_G))

print(f"第3000次迭代的误差结果:")
print(f"  PDE F 方程平均绝对误差: {error_F.item():.4e}")
print(f"  PDE G 方程平均绝对误差: {error_G.item():.4e}")






w_real_series = [s[-4] for s in samples]
w_imag_series = [s[-3] for s in samples]
iterations = np.arange(len(samples))

w_realtensor = torch.tensor(w_real_series)
w_imagtensor = torch.tensor(w_imag_series)
w_realmean = torch.mean(w_realtensor)
w_imagmean = torch.mean(w_imagtensor)
w_realsigma = torch.std(w_realtensor)
w_imagsigma = torch.std(w_imagtensor)




# -------- 图1：随迭代次数变化 --------
plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
plt.plot(iterations, w_real_series, label='Re(w)')
plt.ylabel('w_real')
plt.legend()
plt.subplot(2,1,2)
plt.plot(iterations, w_imag_series, color='orange', label='Im(w)')
plt.xlabel('Iteration')
plt.ylabel('w_imag')
plt.legend()
plt.suptitle('Evolution of w_real and w_imag over HMC iterations')
plt.tight_layout(rect=[0,0,1,0.96])
plt.show()


Leaver_real = 0.85023
Leaver_imag = -0.14365
# -------- 图2：分布条形图（直方图形式）--------
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.hist(w_real_series, bins=30, color='steelblue', edgecolor='black')
plt.axvline(x=Leaver_real, color='red', linestyle='--', linewidth=1.5, label='True Re(w)')
plt.axvline(x=w_realmean, color='blue', linestyle='--', linewidth=1.5, label='mean Re(w)')
plt.axvline(x=w_realmean-w_realsigma, color='blue', linestyle='--', linewidth=1.5, label='1sigma Re(w)')
plt.axvline(x=w_realmean+w_realsigma, color='blue', linestyle='--', linewidth=1.5, label='1sigma Re(w)')
plt.xlabel('w_real')
plt.ylabel('Count')
plt.title('Distribution of Re(w)')
plt.subplot(1,2,2)
plt.hist(w_imag_series, bins=30, color='orange', edgecolor='black')
plt.axvline(x=Leaver_imag, color='red', linestyle='--', linewidth=1.5, label='True Im(w)')
plt.axvline(x=w_imagmean, color='blue', linestyle='--', linewidth=1.5, label='mean Im(w)')
plt.axvline(x=w_imagmean-w_imagsigma, color='blue', linestyle='--', linewidth=1.5, label='1sigma Im(w)')
plt.axvline(x=w_imagmean+w_imagsigma, color='blue', linestyle='--', linewidth=1.5, label='1sigma Im(w)')
plt.xlabel('w_imag')
plt.ylabel('Count')
plt.title('Distribution of Im(w)')
plt.tight_layout()
plt.show()