# Quant-Strat

## Overview

**Quant-Strat** is a personal research repository dedicated to exploring and implementing **quantitative trading strategies**.

The objective of this project is to experiment with **statistical models, machine learning techniques, and financial time-series analysis** while translating quantitative finance concepts into practical implementations using Python.

Each strategy is developed as an **independent research module**, allowing me to explore a specific methodology, test hypotheses, and evaluate performance through systematic backtesting.

For every strategy:

- 📦 **One strategy = one module**
- 🧠 **One quantitative methodology explored**
- 📊 **One backtest with performance evaluation**

Alongside the code, I publish a **LinkedIn post explaining the intuition behind the strategy**, the modeling choices, and the results obtained.

This project therefore acts as both:

- a **quant research playground**
- a **learning environment for quantitative finance**
- a **public log of my progress as a quant researcher**

---

# Research Workflow

Each strategy follows a structured research pipeline:

1. **Hypothesis formulation**  
   Define a market inefficiency or predictive hypothesis.

2. **Data collection & preprocessing**  
   Retrieve financial time-series data and clean it.

3. **Feature engineering**  
   Create predictive variables (technical indicators, statistical features, lagged returns).

4. **Model implementation**  
   Apply statistical models or machine learning algorithms.

5. **Backtesting**  
   Evaluate the strategy using historical data.

6. **Performance analysis**  
   Analyze metrics such as returns, volatility, Sharpe ratio, and drawdown.

---

# First Strategy

## ML XGBoost Return Prediction

This strategy explores the use of **gradient boosted decision trees (XGBoost)** to predict **future asset returns**.

The model uses engineered financial features such as:

- lagged returns
- rolling statistics
- technical indicators

The goal is to determine whether machine learning models can extract **predictive signals from financial time series** and generate trading signals.

The notebook includes:

- data preprocessing
- feature engineering
- model training
- prediction
- strategy backtesting
- performance evaluation

---

# Tech Stack

The project relies primarily on the Python scientific ecosystem:

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**
- **XGBoost**
- **Jupyter Notebooks**

---

# Project Goals

Through this repository I aim to:

- Develop practical experience with **quantitative trading models**
- Implement **machine learning techniques in finance**
- Understand the **statistical properties of financial markets**
- Build a structured **quantitative research workflow**
- Document my progress in **quantitative finance**

---

> ⚠️ **Disclaimer**
> This is a **personal training project** created for **educational purposes** only.
> It is **not intended for real trading**, and **no financial results are guaranteed**.
> The goal is to experiment with **Python, econometrics, and quantitative modeling**.
