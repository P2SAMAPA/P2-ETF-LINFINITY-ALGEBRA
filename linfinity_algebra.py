import numpy as np
from scipy.linalg import eigh

def compute_lead_lag_matrix(returns, max_lag=5):
    """
    Compute lead-lag matrix: average cross-correlation for positive lags minus negative lags? Actually we want asymmetry.
    Let L_ij = mean_{lag>0} corr(r_i(t), r_j(t+lag)) - mean_{lag>0} corr(r_i(t+lag), r_j(t)).
    This gives a skew-symmetric matrix where positive means i leads j.
    """
    n = returns.shape[1]
    lead_lag = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pos_lags = []
            neg_lags = []
            for lag in range(1, max_lag+1):
                # i leads j: corr(r_i(t), r_j(t+lag))
                if len(returns) > lag:
                    pos = np.corrcoef(returns.iloc[:-lag, i], returns.iloc[lag:, j])[0,1]
                    pos_lags.append(pos)
                # j leads i: corr(r_i(t+lag), r_j(t))
                if len(returns) > lag:
                    neg = np.corrcoef(returns.iloc[lag:, i], returns.iloc[:-lag, j])[0,1]
                    neg_lags.append(neg)
            lead_lag[i,j] = np.mean(pos_lags) - np.mean(neg_lags) if pos_lags and neg_lags else 0.0
    return lead_lag

def linfinity_scores(returns, max_order=3, eps=0.1):
    """
    Compute per-ETF L∞ degree using skew-symmetric lead-lag bracket.
    """
    returns_clean = returns.dropna()
    n = returns_clean.shape[1]
    if n < 2:
        return {t: 0.0 for t in returns_clean.columns}
    # Use lead-lag matrix as the 2-bracket (already skew-symmetric)
    bracket2 = compute_lead_lag_matrix(returns_clean, max_lag=5)
    # Invert (I - eps * bracket2)
    I = np.eye(n)
    mat = I - eps * bracket2
    try:
        inv = np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(mat)
    degree = np.linalg.norm(inv, axis=0)
    tickers = returns_clean.columns
    return {ticker: degree[i] for i, ticker in enumerate(tickers)}
