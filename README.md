# Causal Analysis of the 2018 Economic Growth, Regulatory Relief, and Consumer Protection Act

## Overview
This project examines the causal impact of the 2018 Economic Growth, Regulatory Relief, and Consumer Protection Act on U.S. banks. A key element of the Act was raising the Dodd-Frank Act Stress Test (DFAST) threshold from \$50 Billion to \$250 Billion, which exempted many banks from mandatory annual stress testing. The collapse of Silicon Valley Bank in March 2023 underscores the importance of understanding these regulatory changes.

## Hypothesis
The central hypothesis is that banks exempted from the new stress testing requirements exhibit systematically riskier financial profiles compared to banks that remain under the traditional DFAST regime.

## Data Sources
- **FFIEC Uniform Bank Performance Reports (UBPR)**
  - **Frequency:** Annual  
  - **Coverage:** U.S. commercial banks and savings institutions  
  - **Key Variables:** Total loans, loan performance (e.g., non-performing loans, charge-offs), asset size, capital ratios, liabilities, and deposits.

- **Federal Reserve Stress Testing Data (CCAR/DFAST)**
  - **Frequency:** Annual  
  - **Coverage:** Large banks previously subject to stress testing  
  - **Key Variables:** Stress test results, projected capital adequacy, loan loss projections, and regulatory capital requirements.

## Methodology
The analysis employs a difference-in-differences (DiD) framework to compare the financial profiles of banks exempted from the new regulations (treated group) against those that continue to undergo stress testing (control group). Additional analysis, such as a Fuzzy-DiD estimator, may be applied to banks around the \$250 billion asset threshold.

## Authors
- **Ian Bluth** (ib78@duke.edu)
- **Henry Hai** (gh146@duke.edu)

## Date
February 4th, 2025