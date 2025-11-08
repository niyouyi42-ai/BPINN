import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def erf(x):
    """误差函数的近似计算"""
    def integrand(t):
        return np.exp(-t**2)
    
    if isinstance(x, np.ndarray):
        result = np.zeros_like(x)
        for i, xi in enumerate(x):
            if xi == 0:
                result[i] = 0
            elif xi > 0:
                t_vals = np.linspace(0, xi, 1000)
                y_vals = integrand(t_vals)
                result[i] = 2/np.sqrt(np.pi) * np.trapz(y_vals, t_vals)
            else:
                result[i] = -erf(-xi)
        return result
    else:
        if x == 0:
            return 0
        elif x > 0:
            t_vals = np.linspace(0, x, 1000)
            y_vals = integrand(t_vals)
            return 2/np.sqrt(np.pi) * np.trapz(y_vals, t_vals)
        else:
            return -erf(-x)


def diagnose_data(sample_data):
    """诊断数据问题"""
    print("=== 数据诊断报告 ===")
    print(f"数据长度: {len(sample_data)}")
    print(f"数据范围: {np.min(sample_data):.6f} 到 {np.max(sample_data):.6f}")
    print(f"数据均值: {np.mean(sample_data):.6f}")
    print(f"数据标准差: {np.std(sample_data):.6f}")
    print(f"是否有NaN值: {np.isnan(sample_data).any()}")
    print(f"是否有Inf值: {np.isinf(sample_data).any()}")
    
    # 检查概率计算
    sorted_data = np.sort(sample_data)
    n = len(sorted_data)
    empirical_probs = np.arange(1, n + 1) / n
    mu = np.mean(sample_data)
    sigma = np.std(sample_data, ddof=1)
    theoretical_probs = 0.5 * (1 + erf((sorted_data - mu) / (sigma * np.sqrt(2))))
    
    print(f"\n=== 概率计算检查 ===")
    print(f"经验概率范围: {np.min(empirical_probs):.6f} 到 {np.max(empirical_probs):.6f}")
    print(f"理论概率范围: {np.min(theoretical_probs):.6f} 到 {np.max(theoretical_probs):.6f}")
    
    # 检查是否有异常的概率值
    if np.any(theoretical_probs < 0) or np.any(theoretical_probs > 1):
        print("警告: 理论概率超出[0,1]范围!")
    
    return theoretical_probs, empirical_probs

def pp_plot_fixed(sample_data, theoretical_dist='norm', params=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    sorted_data = np.sort(sample_data)
    n = len(sorted_data)
    empirical_probs = np.arange(1, n + 1) / n
    
    if theoretical_dist == 'norm':
        if params is None:
            mu = np.mean(sample_data)
            sigma = np.std(sample_data, ddof=1)
        else:
            mu, sigma = params
        
        theoretical_probs = 0.5 * (1 + erf((sorted_data - mu) / (sigma * np.sqrt(2))))
    
    # 绘制数据点
    ax.scatter(theoretical_probs, empirical_probs, color='blue', alpha=0.6, s=20, label='数据点')
    
    # 绘制参考线
    ax.plot([0, 1], [0, 1], 'r-', linewidth=2, label='完美拟合')
    
    # 设置图形属性
    ax.set_xlabel('理论累积概率')
    ax.set_ylabel('样本累积概率')
    ax.set_title('P-P图 (概率-概率图)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')
    
    return ax

def test_pp_plot():
    
    # 诊断数据
    theoretical_probs, empirical_probs = diagnose_data(w_real_series)
    theoretical_probs, empirical_probs = diagnose_data(w_imag_series)
    
    # 绘制P-P图
    print("\n正在绘制P-P图...")
    plt.figure(figsize=(8, 8))
    pp_plot_fixed(w_real_series)
    plt.show()
    
    print("测试完成!")

# 运行测试
test_pp_plot()