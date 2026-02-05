import numpy as np
import pandas as pd
from dataclasses import dataclass
import matplotlib.pyplot as plt
import os
import glob

DEFAULT_SEEDS = (0, 1, 2, 3, 4)
# =========================
#  Feature map and utilities
# =========================

def squash(z: float) -> float:
    """1-Lipschitz-ish squashing function."""
    
    return 1.0 / (1.0 + np.exp(-z))


def make_phi_fn(d_ctx: int, A: int):
    """
    创建特征映射 φ(x,a) ∈ R^{d_ctx + A}，把上下文和动作 one-hot 拼起来。
    """
    d = d_ctx + A

    def phi(x: np.ndarray, a: int) -> np.ndarray:
        assert x.shape[0] == d_ctx
        e_a = np.zeros(A)
        e_a[a] = 1.0
        feat = np.concatenate([x, e_a])
        assert feat.shape[0] == d
        return feat

    return phi, d


# =========================
#  Config dataclass
# =========================

@dataclass
class Config:
    # Context 和 action 维度
    d_ctx: int = 10
    A: int = 5

    # 总轮数（horizon）
    T: int = 50_000

    # 环境参数
    sigma_theta: float = 1.0
    sigma_noise: float = 0.1

    # Discounted LinUCB 超参（可以根据你论文里的理论微调）
    lambda_reg: float = 1.0
    gamma: float = 0.99
    alpha_ucb: float = 1.0

    # 评估设置
    n_eval: int = 50  # evaluation contexts 数量
    seeds: tuple = DEFAULT_SEEDS

    # 非平稳度 (M, eta) 网格
    M_grid: tuple = (1, 5, 20)
    eta_grid: tuple = (0.0, 0.1, 0.3)

    # teacher 预算用“真实 C”，这里先用比例指定，后面转换成 C = frac * T
    C_grid_frac: tuple = (0.0, 0.05, 0.10, 0.20)

    def __post_init__(self):
        # 派生量
        self.Sigma_x = np.eye(self.d_ctx)
        self.phi, self.d = make_phi_fn(self.d_ctx, self.A)

        # 把比例换成真实预算 C
        self.C_grid = tuple(int(frac * self.T) for frac in self.C_grid_frac)


# =========================
#  Discounted LinUCB learner
# =========================

class DiscountedLinUCB:
    def __init__(self, d: int, A: int, lambda_reg: float, gamma: float, alpha_ucb: float, phi):
        self.d = d
        self.A = A
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.alpha = alpha_ucb
        self.phi = phi

        self.V = lambda_reg * np.eye(d)
        self.b = np.zeros(d)

    def _theta_hat(self) -> np.ndarray:
        # 解 V θ = b
        return np.linalg.solve(self.V, self.b)

    def policy(self, x_t: np.ndarray) -> np.ndarray:
        """
        返回动作分布 π_t(·|x_t)，这里用贪心的 Discounted LinUCB。
        """
        theta_hat = self._theta_hat()
        scores = np.zeros(self.A)
        for a in range(self.A):
            phi_ta = self.phi(x_t, a)
            try:
                V_inv_phi = np.linalg.solve(self.V, phi_ta)
            except np.linalg.LinAlgError:
                V_inv_phi = np.linalg.pinv(self.V) @ phi_ta
            s_ta = np.sqrt(phi_ta @ V_inv_phi)
            m_ta = phi_ta @ theta_hat
            scores[a] = m_ta + self.alpha * s_ta

        a_star = int(np.argmax(scores))
        pi = np.zeros(self.A)
        pi[a_star] = 1.0
        return pi

    def act(self, x_t: np.ndarray):
        """
        选择动作并返回 (a_t, π_t(·|x_t))。
        """
        pi = self.policy(x_t)
        a_t = int(np.argmax(pi))  # 贪心
        return a_t, pi

    def update(self, x_t: np.ndarray, a_t: int, r_tilde: float):
        """
        按论文 Algorithm 2 的折扣更新。
        """
        phi_t = self.phi(x_t, a_t)

        self.V = (
            self.gamma * self.V
            + np.outer(phi_t, phi_t)
            + (1.0 - self.gamma) * self.lambda_reg * np.eye(self.d)
        )
        self.b = self.gamma * self.b + phi_t * r_tilde


# =========================
#  Canonical environment and target policy
# =========================

def sample_theta_c(d: int, sigma_theta: float, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(loc=0.0, scale=sigma_theta, size=d)


def make_pi_dagger_fn(theta_c: np.ndarray, A: int, phi):
    """
    目标策略 π†：在 canonical 环境里对每个 x 贪心。
    """
    def pi_dagger(x: np.ndarray) -> np.ndarray:
        scores = np.zeros(A)
        for a in range(A):
            scores[a] = squash(theta_c @ phi(x, a))
        a_star = int(np.argmax(scores))
        pi = np.zeros(A)
        pi[a_star] = 1.0
        return pi

    return pi_dagger


# =========================
#  Teacher strategies
# =========================

def teacher_none(r_true: np.ndarray):
    """
    无教师基线：learner 直接看到真实 reward。
    """
    r_tilde = r_true.copy()
    cost_t = 0.0
    return r_tilde, cost_t


def teacher_mixture(x_t: np.ndarray,
                    r_true: np.ndarray,
                    theta_c: np.ndarray,
                    lambda_frac: float,
                    A: int,
                    phi):
    """
    Mixture teacher: r̃_t = (1-λ) r_t + λ r_t^c.
    lambda_frac = C / T 控制 teacher 强度。
    """
    r_c = np.zeros(A)
    for a in range(A):
        r_c[a] = squash(theta_c @ phi(x_t, a))

    r_tilde = (1.0 - lambda_frac) * r_true + lambda_frac * r_c
    cost_t = float(lambda_frac)  # 每轮代价上界 λ
    return r_tilde, cost_t


# =========================
#  Single run for one (M, η, C, seed)
# =========================

def run_one_config(cfg: Config,
                   M: int,
                   eta: float,
                   C: int,
                   seed: int,
                   theta_c: np.ndarray,
                   pi_dagger):
    """
    对给定非平稳度 (M, eta)、teacher 预算 C 和随机种子跑一次实验，返回三大指标。
    """
    rng = np.random.default_rng(seed)
    T = cfg.T
    A = cfg.A
    d_ctx = cfg.d_ctx
    phi = cfg.phi

    # 预算 C -> λ = C / T
    lambda_frac = C / T if T > 0 else 0.0

    # 预生成每个 segment 的 θ^(m)
    L = max(1, T // M)
    theta_segments = np.zeros((M, cfg.d))
    theta_segments[0] = rng.normal(loc=0.0, scale=cfg.sigma_theta, size=cfg.d)
    for m in range(1, M):
        delta = rng.normal(loc=0.0, scale=eta, size=cfg.d)
        theta_segments[m] = theta_segments[m-1] + delta

    # 固定 evaluation contexts
    eval_x = rng.multivariate_normal(
        mean=np.zeros(d_ctx),
        cov=cfg.Sigma_x,
        size=cfg.n_eval
    )

    # 初始化 learner
    learner = DiscountedLinUCB(
        d=cfg.d,
        A=A,
        lambda_reg=cfg.lambda_reg,
        gamma=cfg.gamma,
        alpha_ucb=cfg.alpha_ucb,
        phi=phi
    )

    dynreg = 0.0
    cost_tot = 0.0
    mismatch_list = []
    pi_history = []

    for t in range(T):
        # --- 真实环境采样 x_t 和 rewards ---
        seg_idx = min(M - 1, t // L)
        theta_t = theta_segments[seg_idx]

        x_t = rng.multivariate_normal(
            mean=np.zeros(d_ctx),
            cov=cfg.Sigma_x
        )

        r_mean = np.zeros(A)
        r_true = np.zeros(A)
        for a in range(A):
            phi_ta = phi(x_t, a)
            r_mean[a] = squash(theta_t @ phi_ta)
            noise = rng.normal(loc=0.0, scale=cfg.sigma_noise)
            r_true[a] = np.clip(r_mean[a] + noise, 0.0, 1.0)

        # --- teacher 干预 ---
        if C <= 0:
            r_tilde_vec, cost_t = teacher_none(r_true)
        else:
            r_tilde_vec, cost_t = teacher_mixture(
                x_t=x_t,
                r_true=r_true,
                theta_c=theta_c,
                lambda_frac=lambda_frac,
                A=A,
                phi=phi
            )
        cost_tot += cost_t

        # --- learner 选择动作并更新（看到的是 poisoned reward） ---
        a_t, pi_t_x = learner.act(x_t)
        r_tilde_scalar = r_tilde_vec[a_t]
        learner.update(x_t, a_t, r_tilde_scalar)

        # --- 在真实环境上的即时 dynamic regret ---
        best_mean = float(np.max(r_mean))
        dynreg += best_mean - float(r_mean[a_t])

        # --- 在固定 eval_x 上评估 π_t 和 π†，用于 mismatch / stability ---
        pi_eval = np.zeros((cfg.n_eval, A))
        pi_dagger_eval = np.zeros_like(pi_eval)

        for i, x_eval in enumerate(eval_x):
            pi_eval[i] = learner.policy(x_eval)
            pi_dagger_eval[i] = pi_dagger(x_eval)

        pi_history.append(pi_eval)

        mismatch_t = np.mean(
            np.sum(np.abs(pi_eval - pi_dagger_eval), axis=1)
        )
        mismatch_list.append(mismatch_t)

    # 聚合指标
    K = T
    mismatch_K = float(np.mean(mismatch_list))
    dynreg_K = float(dynreg)
    avg_reg = dynreg_K / K

    # Policy stability
    stab_terms = []
    for t in range(1, len(pi_history)):
        diff = np.mean(
            np.sum(np.abs(pi_history[t] - pi_history[t-1]), axis=1)
        )
        stab_terms.append(diff)
    stab_K = float(np.mean(stab_terms)) if stab_terms else 0.0

    return {
        "M": M,
        "eta": eta,
        "C": C,
        "C_frac": lambda_frac,  # 方便你以后分析
        "seed": seed,
        "DynReg": dynreg_K,
        "AvgReg": avg_reg,
        "Mismatch": mismatch_K,
        "Stab": stab_K,
        "CostTot": cost_tot,
        "CostPerStep": cost_tot / T
    }


# =========================
#  Plotting + summary (delegated to analysis.make_plots)
# =========================

def plot_results(csv_path: str, expected_seed_count: int):
    """
    Use the shared plotting/summary module for figures + global table.
    """
    from analysis.make_plots import generate_all

    results_dir = os.path.dirname(os.path.abspath(csv_path))
    generate_all(
        results_dir=results_dir,
        output_dir=results_dir,
        expected_seed_count=expected_seed_count,
        csv_path=csv_path,
    )


# =========================
#  Main sweep script
# =========================

def main():
    # 读取来自 Slurm 的 seed 覆盖（如果存在）
    seed_override = os.environ.get("SEED_OVERRIDE")
    if seed_override is not None:
        seed_override = int(seed_override)

    # ==== 1. 配置超参数（你可以在这里改） ====
    base_seeds = DEFAULT_SEEDS
    if seed_override is not None:
        seeds = (seed_override,)
    else:
        seeds = base_seeds

    cfg = Config(
        d_ctx=5,
        A=5,
        T=50_000,
        sigma_theta=1.0,
        sigma_noise=0.1,
        lambda_reg=1.0,
        gamma=0.99,
        alpha_ucb=1.0,
        n_eval=50,
        seeds=seeds,
        M_grid=(1, 5, 20),
        eta_grid=(0.0, 0.1, 0.3),
        C_grid_frac=(0.0, 0.05, 0.10, 0.20),
    )

    # ==== 2. Canonical θ_c 和 π†（所有 run 共享） ====
    rng_global = np.random.default_rng(12345)
    theta_c = sample_theta_c(cfg.d, cfg.sigma_theta, rng_global)
    pi_dagger = make_pi_dagger_fn(theta_c, cfg.A, cfg.phi)

    # ==== 3. 实验循环 ====
    results = []

    for M in cfg.M_grid:
        for eta in cfg.eta_grid:
            for C in cfg.C_grid:
                for seed in cfg.seeds:
                    print(f"Running M={M}, eta={eta}, C={C}, seed={seed}...")
                    res = run_one_config(
                        cfg=cfg,
                        M=M,
                        eta=eta,
                        C=C,
                        seed=seed,
                        theta_c=theta_c,
                        pi_dagger=pi_dagger
                    )
                    results.append(res)

    df = pd.DataFrame(results)

# 为了避免 job array 互相覆盖输出，每个 seed 单独一个 CSV
    if seed_override is not None:
        csv_path = f"nonstationary_bandit_results_seed{seed_override}.csv"
    else:
        csv_path = "nonstationary_bandit_results.csv"


    df.to_csv(csv_path, index=False)   
    print(f"Saved results to {csv_path}")
    seed_files = sorted(glob.glob("nonstationary_bandit_results_seed*.csv"))
    
    if seed_files:
        print("Found per-seed CSVs:", seed_files)
        dfs = [pd.read_csv(f) for f in seed_files]
        df_all = pd.concat(dfs, ignore_index=True)
        merged_path = "nonstationary_bandit_results_merged.csv"
        df_all.to_csv(merged_path, index=False)
        print(f"Merged to {merged_path}")
        csv_path = merged_path
    else:
        # 如果没有 per-seed 文件，就用单个 CSV（单机模式）
        csv_path = "nonstationary_bandit_results.csv"
        print(f"Using single CSV: {csv_path}")

    # 2) 画图（会直接生成 pdf 图像文件）
    plot_results(csv_path, expected_seed_count=len(seeds))


if __name__ == "__main__":
    main()
