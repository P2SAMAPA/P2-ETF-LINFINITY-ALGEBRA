# L∞‑Algebra Engine for ETFs

Models ETF interactions using L∞‑algebras (homotopy Lie algebras). The engine computes a per‑ETF **Maurer‑Cartan degree** that measures involvement in higher‑order brackets beyond pairwise correlations.

## Features
- Three ETF universes
- Seven rolling windows (63–4536 days)
- 2‑bracket from anti‑symmetrised correlation matrix
- Higher‑bracket contributions via perturbative inversion (I – ε·bracket2)⁻¹
- Score = L2 norm of the Maurer‑Cartan element component
- Best window automatically selected (largest absolute raw signal)
- Two‑tab Streamlit dashboard (auto best + manual window selection)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-linfinity-algebra-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Run training: `python train.py`
3. Launch dashboard: `streamlit run streamlit_app.py`
4. GitHub Actions runs daily.

## Interpretation

- **L∞‑algebras** naturally encode non‑associative, higher‑order interactions.
- A high L∞ degree indicates that the ETF participates in genuine multi‑agent interactions that are not reducible to pairwise relationships.
- Such ETFs may be “structural hubs” in the homotopy sense – potential sources of systemic risk or alpha.

## Requirements

See `requirements.txt`.
