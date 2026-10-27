"""
Market-making simulator: symmetric vs. inventory-aware quoting.

Synthetic random-walk mid-price; exponential fill probability.
Run: python market_maker.py
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Parameters
# ----------------------------------------------------------------------
T          = 1000       # number of time steps
M0         = 100.0      # initial mid-price
SIGMA      = 0.20       # per-step volatility (fraction of price)
H          = 0.10       # half-spread
P0         = 0.50       # fill probability at the mid
K          = 10.0       # fill-decay rate
V          = 10         # shares per fill
FEE        = 0.01       # per-share transaction fee
GAMMA      = 0.02       # inventory-aversion parameter
SEED       = 42

# ----------------------------------------------------------------------
# 2. Price path
# ----------------------------------------------------------------------
def generate_price_path(m0, sigma, T, rng):
    z = rng.standard_normal(T)               # T i.i.d. N(0,1) shocks
    increments = sigma * z                   # per-step moves
    path = m0 + np.concatenate([[0.0], np.cumsum(increments)])
    return path                              # length T+1

# ----------------------------------------------------------------------
# 3. Quoting
# ----------------------------------------------------------------------
def quote(mid, q, h, gamma, inventory_aware):
    if inventory_aware:
        ref = mid - gamma * q
    else:
        ref = mid
    bid = ref - h
    ask = ref + h
    return bid, ask

# ----------------------------------------------------------------------
# 4. Fill probability
# ----------------------------------------------------------------------
def fill_probability(distance, p0, k):
    distance = max(distance, 0.0)            # guard against negative delta
    return p0 * np.exp(-k * distance)

# ----------------------------------------------------------------------
# 5. Main simulation loop
# ----------------------------------------------------------------------
def simulate(path, h, p0, k, v, fee, gamma, inventory_aware, fill_shocks):
    T = len(path) - 1
    q = 0                                    # inventory (shares)
    c = 0.0                                  # cash
    W0 = c + q * path[0]                     # initial wealth

    pnl = np.zeros(T + 1)
    inv = np.zeros(T + 1)
    n_trades = 0

    for t in range(T):
        mid = path[t]

        # --- (a) mark to market at start of step ---
        pnl[t] = c + q * mid - W0
        inv[t] = q

        # --- (b) generate quotes ---
        bid, ask = quote(mid, q, h, gamma, inventory_aware)

        # --- (c) fill probabilities ---
        p_bid = fill_probability(mid - bid, p0, k)
        p_ask = fill_probability(ask - mid, p0, k)

        # --- (d) decide fills using pre-drawn uniforms ---
        u_bid, u_ask = fill_shocks[t]
        bid_filled = u_bid < p_bid
        ask_filled = u_ask < p_ask

        # --- (e) accounting ---
        if bid_filled:
            q += v
            c -= v * bid + v * fee
            n_trades += 1
        if ask_filled:
            q -= v
            c += v * ask - v * fee
            n_trades += 1

    # final mark
    pnl[T] = c + q * path[T] - W0
    inv[T] = q

    return {
        "pnl": pnl,
        "inventory": inv,
        "n_trades": n_trades,
        "final_pnl": pnl[-1],
    }

# ----------------------------------------------------------------------
# 6. Metrics
# ----------------------------------------------------------------------
def metrics(result):
    pnl = result["pnl"]
    inv = result["inventory"]
    running_max = np.maximum.accumulate(pnl)
    drawdown = running_max - pnl
    return {
        "final_pnl": float(pnl[-1]),
        "max_drawdown": float(drawdown.max()),
        "avg_abs_inventory": float(np.mean(np.abs(inv))),
        "n_trades": int(result["n_trades"]),
        "pnl_vol": float(np.std(np.diff(pnl))),
    }

# ----------------------------------------------------------------------
# 7. Experiment runner
# ----------------------------------------------------------------------
def run():
    rng_price = np.random.default_rng(SEED)
    path = generate_price_path(M0, SIGMA, T, rng_price)

    # Same fill random numbers for both strategies
    rng_fill = np.random.default_rng(SEED + 1)
    fill_shocks = rng_fill.random((T, 2))    # (u_bid, u_ask) per step

    res_naive = simulate(path, H, P0, K, V, FEE, GAMMA,
                         inventory_aware=False, fill_shocks=fill_shocks)
    res_inv   = simulate(path, H, P0, K, V, FEE, GAMMA,
                         inventory_aware=True,  fill_shocks=fill_shocks)

    print(f"{'metric':<22}{'naive':>12}{'inventory-aware':>20}")
    for key in metrics(res_naive):
        a = metrics(res_naive)[key]
        b = metrics(res_inv)[key]
        print(f"{key:<22}{a:>12.4f}{b:>20.4f}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(res_naive["pnl"], label="naive")
    axes[0].plot(res_inv["pnl"],   label="inventory-aware")
    axes[0].set_ylabel("PnL")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(res_naive["inventory"], label="naive")
    axes[1].plot(res_inv["inventory"],   label="inventory-aware")
    axes[1].set_ylabel("inventory (shares)")
    axes[1].set_xlabel("step")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("comparison.png", dpi=120)
    plt.show()

if __name__ == "__main__":
    run()