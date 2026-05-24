import numpy as np
from scipy.linalg import eigh

def compute_2_bracket(returns):
    """Binary bracket: anti‑symmetrised correlation."""
    corr = returns.corr().values
    # 2‑bracket = (corr - corr.T) / 2  (anti‑symmetrised)
    bracket2 = (corr - corr.T) / 2.0
    return bracket2

def compute_higher_bracket_contributions(bracket2, max_order=3, eps=0.1):
    """
    Simulate higher‑bracket contributions via perturbative expansion:
    The Maurer‑Cartan element mc satisfies d(mc) + ½[mc, mc] + ... = 0.
    Here we treat the 2‑bracket as the leading term and compute the "degree"
    of each basis element (ETF) in the solution.
    We approximate: degree_i = ||(I - eps * bracket2)^(-1) e_i||_2
    """
    n = bracket2.shape[0]
    # Identity matrix
    I = np.eye(n)
    # Invert (I - eps * bracket2) – this captures the effect of higher brackets
    try:
        mat = I - eps * bracket2
        inv = np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(mat)
    # Degree = L2 norm of each column (or row) – contribution to Maurer‑Cartan element
    degree = np.linalg.norm(inv, axis=0)
    return degree

def linfinity_scores(returns, max_order=3, eps=0.1):
    """
    Compute per‑ETF L∞ degree: measures how much an ETF participates in
    higher‑order homotopy brackets (beyond pairwise correlations).
    """
    returns_clean = returns.dropna()
    n = returns_clean.shape[1]
    if n < 2:
        return {t: 0.0 for t in returns_clean.columns}
    bracket2 = compute_2_bracket(returns_clean)
    degree = compute_higher_bracket_contributions(bracket2, max_order, eps)
    tickers = returns_clean.columns
    return {ticker: degree[i] for i, ticker in enumerate(tickers)}
