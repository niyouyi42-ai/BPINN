import numpy as np
import torch


class HMC:
    def __init__(
        self,
        target_log_prob_fn,
        init_state,
        num_results=1000,
        num_burnin=1000,
        num_leapfrog_steps=30,
        step_size=0.1,
    ):
        """
        PyTorch 实现的 Hamiltonian Monte Carlo (HMC)

        参数:
        - target_log_prob_fn: 目标分布的 log 概率函数, 输入 torch.Tensor, 输出标量
        - init_state: 初始状态 (array-like)
        - num_results: 采样数量 (去掉 burn-in 后)
        - num_burnin: burn-in 步数
        - num_leapfrog_steps: leapfrog 步数
        - step_size: 步长
        """
        self.target_log_prob_fn = target_log_prob_fn
        self.current_state = torch.tensor(init_state, dtype=torch.float32, requires_grad=True)
        self.num_results = num_results
        self.num_burnin = num_burnin
        self.num_leapfrog_steps = num_leapfrog_steps
        self.step_size = step_size

    def leapfrog(self, position, momentum):
        """leapfrog 积分"""
        position = position.clone().detach().requires_grad_(True)
        momentum = momentum.clone().detach()
        step_size = self.step_size

        # 半步更新动量
        potential = -self.target_log_prob_fn(position)
        grad_potential = torch.autograd.grad(potential, position)[0]
        momentum = momentum - 0.5 * step_size * grad_potential

        # 循环迭代
        for i in range(self.num_leapfrog_steps):
            # 更新位置
            position = position + step_size * momentum
            position.requires_grad_(True)

            if i != self.num_leapfrog_steps - 1:
                potential = -self.target_log_prob_fn(position)
                grad_potential = torch.autograd.grad(potential, position)[0]
                momentum = momentum - step_size * grad_potential

        # 最后半步更新动量
        potential = -self.target_log_prob_fn(position)
        grad_potential = torch.autograd.grad(potential, position)[0]
        momentum = momentum - 0.5 * step_size * grad_potential

        # 反转动量 (保证对称性)
        momentum = -momentum

        return position.detach(), momentum.detach()

    def run_chain(self):
        samples = []
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
            acceptance_prob = torch.exp(current_hamiltonian - proposed_hamiltonian).clamp(max=1.0)
            if torch.rand(1) < acceptance_prob:
                current_position = proposed_position.clone().detach().requires_grad_(True)
                if i >= self.num_burnin:
                    accept_count += 1
            else:
                current_position = current_position.clone().detach().requires_grad_(True)

            if i >= self.num_burnin:
                samples.append(current_position.detach().numpy())

        acceptance_rate = accept_count / self.num_results
        samples = np.array(samples)
        return samples, acceptance_rate