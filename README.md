\# Market Making Simulator: Naive vs. Inventory-Aware Quoting



A Python simulation demonstrating market making on a synthetic random-walk order book. It compares a naive symmetric quoting strategy against an inventory-aware strategy.



\## The Strategy

Market makers provide liquidity by quoting a bid and an ask around a fair value, earning the spread. However, they face \*\*inventory risk\*\*: if they accumulate too much long or short inventory, a shift in the underlying price will wipe out their spread profits.



This script simulates a mid-price path using a random walk, calculates exponential fill probabilities based on quote distance, and runs two strategies:



1\. \*\*Naive Quoting:\*\* Quotes symmetrically around the mid-price.

2\. \*\*Inventory-Aware Quoting:\*\* Skews the quotes based on current inventory (similar to the Avellaneda-Stoikov framework). As inventory builds up, the strategy shifts both quotes lower to discourage further buying and encourage selling.



\## Features

\- \*\*Random Walk Mid-price:\*\* Synthetic data generation using a configurable per-step volatility (`SIGMA`).

\- \*\*Exponential Fill Probabilities:\*\* Fill probability decays exponentially the further the quote is from the mid-price.

\- \*\*Transaction Fees:\*\* Deducts a per-share fee (`FEE`) on every trade.

\- \*\*Risk Metrics:\*\* Tracks Max Drawdown, Average Absolute Inventory, PnL Volatility, and Final PnL.

\- \*\*Shared Random Shocks:\*\* Uses the exact same fill shocks (random uniforms) for both strategies to ensure a fair backtest comparison.



\## How to Run

1\. Install the required dependencies:

```bash

pip install numpy matplotlib

