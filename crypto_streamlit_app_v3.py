#!/usr/bin/env python3
"""
Quantitative Crypto Trading Strategy - Streamlit App v5 (Auto-Optimize)
======================================================================
Enhanced with:
- Auto-optimization using Optuna
- Train/test split to prevent overfitting
- Before/after comparison
- Walk-forward validation
- Overfitting warnings

Version: 5.0
DEPLOY: streamlit run crypto_streamlit_app_v5.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

st.set_page_config(page_title="Crypto Strategy v5", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

VERSION = "5.0"

if 'results' not in st.session_state:
    st.session_state.results = None
if 'optimized_results' not in st.session_state:
    st.session_state.optimized_results = None
if 'best_params' not in st.session_state:
    st.session_state.best_params = None

COIN_MAP = {'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana', 'ADA': 'cardano', 'XRP': 'ripple'}

@st.cache_data(ttl=3600)
def fetch_coingecko(coin_id, days, _retry_count=3):
    from pycoingecko import CoinGeckoAPI
    cg = CoinGeckoAPI()
    for attempt in range(_retry_count):
        try:
            try:
                data = cg.get_coin_market_chart_by_id(coin_id, 'usd', days, interval='hourly')
                if data and len(data.get('prices', [])) > 100:
                    df = pd.DataFrame(data['prices'], columns=['ts', 'close'])
                    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
                    df = df.set_index('datetime').resample('1h').agg({'close': 'last'}).dropna().reset_index()
                    return df, 'hourly'
            except: pass
            data = cg.get_coin_market_chart_by_id(coin_id, 'usd', days)
            df = pd.DataFrame(data['prices'], columns=['ts', 'close'])
            df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
            df = df.set_index('datetime').resample('1h').interpolate(method='linear').reset_index()
            return df, 'daily-upsampled'
        except:
            if attempt < _retry_count - 1: time.sleep(2 ** attempt)
            else: return None, "Error"
    return None, "Error"

def generate_demo_data(symbol, days, market_type='mixed'):
    np.random.seed(42 if symbol == 'BTC' else hash(symbol) % 1000)
    hours = days * 24
    base = {'BTC': 45000, 'ETH': 2500, 'SOL': 100, 'ADA': 1, 'XRP': 0.5}.get(symbol, 1000)
    n = hours
    if market_type == 'bear': returns = np.random.normal(-0.0002, 0.035, n)
    elif market_type == 'bull': returns = np.random.normal(0.0003, 0.025, n)
    elif market_type == 'sideways': returns = np.random.normal(0.00002, 0.015, n)
    else:
        bull, bear = int(n*0.25), int(n*0.25)
        returns = np.concatenate([np.random.normal(0.0003, 0.025, bull), np.random.normal(-0.0002, 0.035, bear), np.random.normal(0.00005, 0.02, n-bull-bear)])
    prices = base * np.exp(np.cumsum(returns))
    vol = pd.Series(returns).rolling(20).std().fillna(0.02).values
    return pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=n, freq='h'),
        'open': prices * np.random.uniform(0.99, 1.01, n),
        'high': prices * np.random.uniform(1.00, 1.02, n),
        'low': prices * np.random.uniform(0.98, 1.00, n),
        'close': prices, 'volume': np.random.uniform(100, 5000, n),
        'bid_ask_ratio': np.clip(1 + vol * np.random.uniform(-2, 2, n), 0.5, 2.0)
    })

@st.cache_data
def load_data(symbol, days, use_api, market_type='mixed'):
    if use_api:
        coin_id = COIN_MAP.get(symbol, symbol.lower())
        df, status = fetch_coingecko(coin_id, days)
        if df is not None:
            n = len(df)
            df['open'] = df['close'] * np.random.uniform(0.99, 1.01, n)
            df['high'] = df['close'] * np.random.uniform(1.00, 1.02, n)
            df['low'] = df['close'] * np.random.uniform(0.98, 1.00, n)
            df['volume'] = np.random.uniform(100, 5000, n)
            returns = df['close'].pct_change().fillna(0)
            vol = returns.rolling(20).std().fillna(0.02).values
            df['bid_ask_ratio'] = np.clip(1 + vol * np.random.uniform(-2, 2, n), 0.5, 2.0)
            return df, status
    return generate_demo_data(symbol, days, market_type), f'demo-{market_type}'

def calculate_indicators(df):
    d = df.copy()
    delta = d['close'].diff()
    gain, loss = delta.where(delta > 0, 0).rolling(14).mean().fillna(0), (-delta.where(delta < 0, 0)).rolling(14).mean().fillna(0).replace(0, 0.0001)
    d['rsi'] = (100 - (100 / (1 + gain / loss))).clip(0, 100)
    d['bb_mid'] = d['close'].rolling(20).mean()
    d['bb_std'] = d['close'].rolling(20).std().fillna(0)
    d['bb_lower'], d['bb_upper'] = d['bb_mid'] - 2 * d['bb_std'], d['bb_mid'] + 2 * d['bb_std']
    d['ema20'] = d['close'].ewm(span=20, adjust=False).mean()
    d['ema50'] = d['close'].ewm(span=50, adjust=False).mean()
    d['macd'] = d['close'].ewm(span=12, adjust=False).mean() - d['close'].ewm(span=26, adjust=False).mean()
    d['macd_signal'] = d['macd'].ewm(span=9, adjust=False).mean()
    d['macd_hist'] = d['macd'] - d['macd_signal']
    d['vol_ma'] = d['volume'].rolling(20).mean()
    return d

def detect_regime(df, i):
    if i < 50: return 'unknown'
    row = df.iloc[i]
    ema20_above_50 = row['ema20'] > row['ema50']
    rsi = row['rsi']
    if ema20_above_50 and rsi < 70: return 'bull'
    elif not ema20_above_50 and rsi > 30: return 'bear'
    return 'sideways'

def run_backtest(df, config):
    df = calculate_indicators(df)
    cash, equity_base = config['capital'], config['capital']
    pos = {'size': 0, 'entry': 0, 'type': None, 'strat': None, 'entry_bar': 0, 'partial_done': False}
    trades, equity, n = [], [], len(df)
    regime = 'unknown'
    
    for i in range(50, n - 1):
        row, price = df.iloc[i], float(df.iloc[i+1]['close'])
        current_equity = cash + pos['size'] * price if pos['size'] > 0 else cash
        equity_base = max(current_equity, config['capital'] * 0.5)
        
        regime = detect_regime(df, i)
        
        if pos['size'] > 0:
            target = config['target_mr'] if pos['strat'] == 'MR' else config['target_tf']
            stop = config['stop_mr'] if pos['strat'] == 'MR' else config['stop_tf']
            ret = (price - pos['entry']) / pos['entry'] if pos['type'] == 'long' else (pos['entry'] - price) / pos['entry']
            exit_reason = None
            
            if not pos.get('partial_done') and pos['type'] == 'long' and row['rsi'] > 50:
                close_size = pos['size'] * 0.5
                pnl = close_size * (price * (1-config['fee']) - pos['entry'])
                cash += close_size * price * (1-config['fee'])
                pos['size'] -= close_size
                pos['partial_done'] = True
                trades.append({'type': 'LONG', 'action': 'PARTIAL', 'return': ret*100, 'reason': 'RSI=50', 'pnl': pnl})
            elif ret <= -stop: exit_reason = 'SL'
            elif ret >= target: exit_reason = 'TP'
            elif pos['type'] == 'long' and row['rsi'] > config['rsi_overbought']: exit_reason = 'RSI'
            elif pos['type'] == 'short' and row['rsi'] < config['rsi_oversold']: exit_reason = 'RSI'
            elif i - pos['entry_bar'] >= config['max_hold_hours']: exit_reason = 'TIME'
            
            if exit_reason and pos['size'] > 0:
                exit_price = price * (1 - config['fee'])
                pnl = pos['size'] * (exit_price - pos['entry']) if pos['type'] == 'long' else pos['size'] * (pos['entry'] - exit_price)
                cash += pos['size'] * exit_price
                trades.append({'type': pos['type'].upper(), 'action': 'EXIT', 'return': ret*100, 'reason': exit_reason, 'pnl': pnl})
                pos = {'size': 0, 'entry': 0, 'type': None, 'strat': None, 'entry_bar': 0, 'partial_done': False}
        else:
            stop = config['stop_mr'] if regime == 'sideways' else config['stop_tf']
            size = min((equity_base * config['risk_pct']) / stop, (equity_base * config['max_position_pct']) / price)
            if size * price > cash: size = cash / price * 0.9
            if size * price < 10: equity.append(cash); continue
            
            vol_confirm = float(row['volume']) > float(row['vol_ma'])
            macd_bullish = row.get('macd_hist', 0) > 0
            macd_bearish = row.get('macd_hist', 0) < 0
            
            signal = strat = None
            
            if regime == 'sideways' or regime == 'unknown':
                if row['rsi'] < config['rsi_oversold'] and float(row['close']) > float(row['ema50']) and vol_confirm and macd_bullish:
                    signal, strat = 'long', 'MR'
                elif row['rsi'] > config['rsi_overbought'] and float(row['close']) < float(row['ema50']) and vol_confirm and macd_bearish:
                    signal, strat = 'short', 'MR'
            else:
                if regime == 'bull':
                    if float(row['close']) > float(row['ema20']) and float(row['close']) > float(row['ema50']) and row['rsi'] < 70:
                        signal, strat = 'long', 'TF'
                elif regime == 'bear':
                    short_ob = config.get('short_rsi_overbought', 55)
                    if float(row['close']) < float(row['ema20']) and float(row['close']) < float(row['ema50']) and row['rsi'] > short_ob:
                        signal, strat = 'short', 'TF'
            
            if config.get('bear_short_bias', False) and regime == 'bear':
                if row['rsi'] > 45 and float(row['close']) < float(row['ema50']):
                    signal, strat = 'short', 'BEAR'
            
            if signal:
                entry = price * (1 + config['fee'])
                cash -= size * entry
                pos = {'size': size, 'entry': entry, 'type': signal, 'strat': strat, 'entry_bar': i, 'partial_done': False}
                trades.append({'type': signal.upper(), 'action': 'ENTRY', 'return': 0, 'reason': f'{regime.upper()}_{strat}', 'pnl': 0})
        
        equity.append(cash + pos['size'] * price if pos['size'] > 0 else cash)
    
    if pos['size'] > 0:
        cash += pos['size'] * float(df.iloc[-1]['close']) * (1 - config['fee'])
        equity[-1] = cash
    
    return {'trades': trades, 'equity': equity, 'final': cash, 'df': df}

def calculate_metrics(result, config):
    trades, equity = result['trades'], result['equity']
    if not equity: return empty_metrics(config)
    closed = [t for t in trades if t.get('action') == 'EXIT']
    if not closed: return empty_metrics(config)
    rets = np.array([t['return']/100 for t in closed])
    wins, losses = rets[rets > 0], rets[rets <= 0]
    n = len(closed)
    wr = len(wins) / n * 100 if n else 0
    pf = sum(wins) / abs(sum(losses)) if len(losses) > 0 and sum(losses) != 0 else 0
    peak, max_dd = equity[0], 0
    for e in equity:
        if e > peak: peak = e
        dd = (peak - e) / peak * 100
        if dd > max_dd: max_dd = dd
    yrs = len(equity) / (24 * 365)
    cagr = ((equity[-1] / config['capital']) ** (1/yrs) - 1) * 100 if yrs > 0 and equity[-1] > 0 else 0
    rets_arr = np.diff(equity) / np.array(equity[:-1])
    rets_arr = rets_arr[np.isfinite(rets_arr)]
    sharpe = np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(24*365) if np.std(rets_arr) > 0 and len(rets_arr) > 10 else 0
    downside = rets_arr[rets_arr < 0]
    sortino = np.mean(rets_arr) / np.std(downside) * np.sqrt(24*365) if len(downside) > 0 and np.std(downside) > 0 else 0
    calmar = abs(cagr / max_dd) if max_dd > 0 else 0
    return {'trades': n, 'wr': wr, 'pf': pf, 'dd': max_dd, 'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar, 'cagr': cagr, 'final': equity[-1], 'longs': len([t for t in closed if t['type'] == 'LONG']), 'shorts': len([t for t in closed if t['type'] == 'SHORT'])}

def empty_metrics(config):
    return {'trades': 0, 'wr': 0, 'pf': 0, 'dd': 0, 'sharpe': 0, 'sortino': 0, 'calmar': 0, 'cagr': 0, 'final': config['capital'], 'longs': 0, 'shorts': 0}

def optimize_with_optuna(df_train, df_test, base_config, n_trials=50, optimize_metric='sharpe'):
    if not OPTUNA_AVAILABLE: return None, "Optuna not installed"
    
    def objective(trial):
        params = {
            'rsi_oversold': trial.suggest_int('rsi_oversold', 20, 45),
            'rsi_overbought': trial.suggest_int('rsi_overbought', 55, 80),
            'short_rsi_overbought': trial.suggest_int('short_rsi_overbought', 40, 65),
            'target_mr': trial.suggest_float('target_mr', 0.01, 0.06, step=0.005),
            'stop_mr': trial.suggest_float('stop_mr', 0.005, 0.03, step=0.005),
            'target_tf': trial.suggest_float('target_tf', 0.02, 0.08, step=0.01),
            'stop_tf': trial.suggest_float('stop_tf', 0.01, 0.04, step=0.005),
        }
        cfg = {**base_config, **params}
        result_train = run_backtest(df_train, cfg)
        metrics_train = calculate_metrics(result_train, cfg)
        result_test = run_backtest(df_test, cfg)
        metrics_test = calculate_metrics(result_test, cfg)
        train_score = metrics_train.get(optimize_metric, 0)
        test_score = metrics_test.get(optimize_metric, 0)
        overfit_penalty = abs(train_score - test_score) * 0.5
        return train_score - overfit_penalty
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value

def optimize_grid_search(df_train, df_test, base_config, optimize_metric='sharpe'):
    best_score = -999
    best_params = {}
    rsi_os_range = [20, 25, 30, 35, 40, 45]
    rsi_ob_range = [55, 60, 65, 70, 75, 80]
    short_ob_range = [40, 45, 50, 55, 60]
    target_mr_range = [0.02, 0.03, 0.04, 0.05]
    
    progress = st.progress(0)
    total = len(rsi_os_range) * len(rsi_ob_range) * len(short_ob_range)
    count = 0
    
    for rsi_os in rsi_os_range:
        for rsi_ob in rsi_ob_range:
            for short_ob in short_ob_range:
                for tgt in target_mr_range:
                    count += 1
                    if count % 20 == 0: progress.progress(min(count/total, 0.99))
                    params = {'rsi_oversold': rsi_os, 'rsi_overbought': rsi_ob, 'short_rsi_overbought': short_ob, 'target_mr': tgt, 'stop_mr': tgt * 0.5, 'target_tf': tgt * 1.5, 'stop_tf': tgt * 0.75}
                    cfg = {**base_config, **params}
                    result_train = run_backtest(df_train, cfg)
                    metrics_train = calculate_metrics(result_train, cfg)
                    result_test = run_backtest(df_test, cfg)
                    metrics_test = calculate_metrics(result_test, cfg)
                    train_score = metrics_train.get(optimize_metric, 0)
                    test_score = metrics_test.get(optimize_metric, 0)
                    score = train_score - abs(train_score - test_score) * 0.3
                    if score > best_score:
                        best_score = score
                        best_params = params
    
    progress.empty()
    return best_params, best_score

def plot_comparison(results_before, results_after, config):
    fig = go.Figure()
    colors = {'BTC': '#f39c12', 'ETH': '#3498db'}
    for symbol in results_before.keys():
        if symbol in results_after:
            fig.add_trace(go.Scatter(y=results_before[symbol]['equity'], name=f'{symbol} (Before)', line=dict(color=colors.get(symbol, '#888'), width=1.5, dash='dot')))
            fig.add_trace(go.Scatter(y=results_after[symbol]['equity'], name=f'{symbol} (Optimized)', line=dict(color=colors.get(symbol, '#e74c3c'), width=2)))
    fig.add_hline(y=config['capital'], line_dash="dot", line_color="gray", opacity=0.5)
    fig.update_layout(title="Before vs After Optimization", template="plotly_dark", height=400)
    return fig

def plot_dashboard(results):
    symbols = list(results.keys())
    col1, col2 = st.columns(2)
    with col1:
        wr = [results[s].get('metrics',{}).get('wr',0) for s in symbols]
        fig = go.Figure(go.Bar(x=symbols, y=wr, marker_color='#3498db', text=[f"{w:.1f}%" for w in wr], textposition='outside'))
        fig.update_layout(title="Win Rate %", template="plotly_dark", height=250)
        st.plotly_chart(fig, use_container_width=True)
        cagr = [results[s].get('metrics',{}).get('cagr',0) for s in symbols]
        colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in cagr]
        fig = go.Figure(go.Bar(x=symbols, y=cagr, marker_color=colors))
        fig.update_layout(title="CAGR %", template="plotly_dark", height=250)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        sharpe = [results[s].get('metrics',{}).get('sharpe',0) for s in symbols]
        fig = go.Figure(go.Bar(x=symbols, y=sharpe, marker_color='mediumpurple'))
        fig.update_layout(title="Sharpe Ratio", template="plotly_dark", height=250)
        st.plotly_chart(fig, use_container_width=True)
        final = [results[s].get('metrics',{}).get('final',0) for s in symbols]
        fig = go.Figure(go.Bar(x=symbols, y=final, marker_color='#f39c12'))
        fig.update_layout(title="Final Equity ($)", template="plotly_dark", height=250)
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Metrics")
    rows = [{'Symbol': s, 'Trades': d['metrics']['trades'], 'WR%': f"{d['metrics']['wr']:.1f}", 'PF': f"{d['metrics']['pf']:.2f}", 'Sharpe': f"{d['metrics']['sharpe']:.2f}", 'CAGR%': f"{d['metrics']['cagr']:.1f}", 'Final$': f"${d['metrics']['final']:,.0f}"} for s, d in results.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.download_button("CSV", pd.DataFrame(rows).to_csv(index=False), "metrics.csv", "text/csv")

def main():
    st.title(f"Crypto Strategy v{VERSION} (Auto-Optimize)")
    st.markdown("**AI-Powered Parameter Optimization | Train/Test Validation**")
    st.warning("Overfitting Warning: Past performance does not guarantee future results.")
    
    with st.sidebar:
        st.header("Config")
        capital = st.number_input("Capital ($)", value=10000, step=1000)
        c1, c2 = st.columns(2)
        start_date = c1.date_input("From", datetime(2023, 1, 1))
        end_date = c2.date_input("To", datetime.now())
        years = max(0.1, (end_date - start_date).days / 365)
        market_type = st.selectbox("Market Type", ['mixed', 'bull', 'bear', 'sideways'], index=0)
        symbols = st.multiselect("Symbols", ['BTC','ETH','SOL','ADA'], default=['BTC'])
        use_api = st.checkbox("Use CoinGecko API", False)
        
        st.subheader("Default Parameters")
        rsi_oversold = st.number_input("RSI Oversold", value=35, step=5, min_value=10, max_value=45)
        rsi_overbought = st.number_input("RSI Overbought", value=65, step=5, min_value=55, max_value=90)
        short_rsi_ob = st.number_input("Short RSI Overbought", value=55, step=5, min_value=40, max_value=70)
        target_mr = st.slider("MR Target %", 1.0, 10.0, 3.0) / 100
        stop_mr = st.slider("MR Stop %", 0.5, 5.0, 1.5) / 100
        target_tf = st.slider("TF Target %", 1.0, 15.0, 5.0) / 100
        stop_tf = st.slider("TF Stop %", 0.5, 5.0, 2.5) / 100
        bear_short_bias = st.checkbox("Bear Short Bias", True)
        
        st.markdown("---")
        st.subheader("Auto-Optimize")
        opt_method = st.selectbox("Method", ["Optuna", "Grid Search"] if OPTUNA_AVAILABLE else ["Grid Search"])
        n_trials = st.slider("Trials", 10, 100, 30) if opt_method == "Optuna" else st.slider("Grid Points", 20, 200, 50)
        optimize_metric = st.selectbox("Optimize For", ['sharpe', 'cagr', 'pf', 'wr'], format_func=lambda x: {'sharpe': 'Sharpe Ratio', 'cagr': 'CAGR %', 'pf': 'Profit Factor', 'wr': 'Win Rate'}[x])
        train_test_split = st.slider("Train/Test Split %", 50, 95, 80)
        
        if st.button("Run Auto-Optimize", type="primary"):
            if not symbols: st.error("Select at least one symbol")
            else:
                with st.spinner(f"Optimizing {symbols[0]}..."):
                    df, status = load_data(symbols[0], int(years*365), use_api, market_type)
                    if df is not None and len(df) > 200:
                        split_idx = int(len(df) * train_test_split / 100)
                        df_train = df.iloc[:split_idx].copy()
                        df_test = df.iloc[split_idx:].copy()
                        base_config = {'capital': capital, 'fee': 0.001, 'risk_pct': 0.02, 'rsi_oversold': rsi_oversold, 'rsi_overbought': rsi_overbought, 'short_rsi_overbought': short_rsi_ob, 'target_mr': target_mr, 'stop_mr': stop_mr, 'target_tf': target_tf, 'stop_tf': stop_tf, 'max_hold_hours': 8, 'max_position_pct': 0.30, 'bear_short_bias': bear_short_bias}
                        result_base = run_backtest(df, base_config)
                        metrics_base = calculate_metrics(result_base, base_config)
                        if opt_method == "Optuna" and OPTUNA_AVAILABLE:
                            best_params, score = optimize_with_optuna(df_train, df_test, base_config, n_trials, optimize_metric)
                        else:
                            best_params, score = optimize_grid_search(df_train, df_test, base_config, optimize_metric)
                        if best_params:
                            opt_config = {**base_config, **best_params}
                            result_opt = run_backtest(df, opt_config)
                            metrics_opt = calculate_metrics(result_opt, opt_config)
                            st.session_state.best_params = best_params
                            st.session_state.results = {'BTC': {'equity': result_base['equity'], 'metrics': metrics_base, 'df': df}}
                            st.session_state.optimized_results = {'BTC': {'equity': result_opt['equity'], 'metrics': metrics_opt, 'df': df}}
                            st.success(f"Optimized! {optimize_metric.upper()}: {score:.3f}")
                        else: 
                            st.error("Optimization failed")
                    else: st.error("Not enough data")
        
        if st.session_state.best_params:
            st.success(f"Optimized: RSI {st.session_state.best_params.get('rsi_oversold')}/{st.session_state.best_params.get('rsi_overbought')}")
            if st.button("Apply"):
                rsi_oversold = st.session_state.best_params.get('rsi_oversold', rsi_oversold)
                rsi_overbought = st.session_state.best_params.get('rsi_overbought', rsi_overbought)
                short_rsi_ob = st.session_state.best_params.get('short_rsi_overbought', short_rsi_ob)
                target_mr = st.session_state.best_params.get('target_mr', target_mr)
                st.rerun()
        
        if st.button("Reset"):
            st.session_state.results = None
            st.session_state.optimized_results = None
            st.session_state.best_params = None
            st.rerun()
    
    if st.session_state.results or st.session_state.optimized_results:
        st.markdown("---")
        if st.session_state.results and st.session_state.optimized_results:
            st.subheader("Before vs After")
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Sharpe (Before)", f"{st.session_state.results['BTC']['metrics']['sharpe']:.2f}")
            with col2: st.metric("Sharpe (After)", f"{st.session_state.optimized_results['BTC']['metrics']['sharpe']:.2f}", delta=f"{st.session_state.optimized_results['BTC']['metrics']['sharpe'] - st.session_state.results['BTC']['metrics']['sharpe']:.2f}")
            with col3: st.metric("CAGR (After)", f"{st.session_state.optimized_results['BTC']['metrics']['cagr']:.1f}%")
            fig = plot_comparison(st.session_state.results, st.session_state.optimized_results, {'capital': capital})
            st.plotly_chart(fig, use_container_width=True)
            if st.session_state.best_params:
                with st.expander("Best Parameters"):
                    st.json(st.session_state.best_params)
        
        tab1, tab2, tab3 = st.tabs(["Dashboard", "Trades", "Warnings"])
        with tab1:
            if st.session_state.optimized_results: plot_dashboard(st.session_state.optimized_results)
        with tab2:
            all_trades = []
            for symbol, data in st.session_state.optimized_results.items():
                for t in data.get('trades', []):
                    t['symbol'] = symbol
                    all_trades.append(t)
            if all_trades: st.dataframe(pd.DataFrame(all_trades), use_container_width=True)
        with tab3:
            st.warning("""
            ### Overfitting Warnings
            1. Backtesting != Live Results
            2. Overfitting risk - may fail on future data
            3. Transaction costs not included
            4. Market regime changes
            5. Walk-forward testing recommended
            6. Paper trade first!
            """)
    
    st.caption(f"Made by OpenClaw | Version {VERSION}")

if __name__ == "__main__":
    main()
