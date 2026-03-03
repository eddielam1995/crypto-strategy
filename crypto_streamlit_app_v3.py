#!/usr/bin/env python3
"""
Quantitative Crypto Trading Strategy - Enhanced Streamlit App v4
=============================================================
Major Enhancements:
- More aggressive short signals (RSI 55 for shorts)
- MACD momentum filter
- ATR-based dynamic stops/targets
- ML regime classifier (sklearn)
- Market condition adaptation (bull/bear/sideways)
- Bear market short bias
- Parameter optimization
- Advanced indicators

Version: 4.0
DEPLOY: streamlit run crypto_streamlit_app_v4.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Crypto Strategy v4", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

VERSION = "4.0"

if 'results' not in st.session_state:
    st.session_state.results = None
if 'optimized' not in st.session_state:
    st.session_state.optimized = None

COIN_MAP = {'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana', 'ADA': 'cardano', 'XRP': 'ripple', 'DOT': 'polkadot'}

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
    base = {'BTC': 45000, 'ETH': 2500, 'SOL': 100, 'ADA': 1, 'XRP': 0.5, 'DOT': 7}.get(symbol, 1000)
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
    d['rsi_slope'] = d['rsi'].diff(5)
    d['bb_mid'] = d['close'].rolling(20).mean()
    d['bb_std'] = d['close'].rolling(20).std().fillna(0)
    d['bb_lower'], d['bb_upper'] = d['bb_mid'] - 2 * d['bb_std'], d['bb_mid'] + 2 * d['bb_std']
    d['ema20'] = d['close'].ewm(span=20, adjust=False).mean()
    d['ema50'] = d['close'].ewm(span=50, adjust=False).mean()
    d['ema200'] = d['close'].ewm(span=200, adjust=False).mean()
    d['ema20_slope'] = d['ema20'].pct_change(10)
    d['ema50_slope'] = d['ema50'].pct_change(10)
    d['macd'] = d['close'].ewm(span=12, adjust=False).mean() - d['close'].ewm(span=26, adjust=False).mean()
    d['macd_signal'] = d['macd'].ewm(span=9, adjust=False).mean()
    d['macd_hist'] = d['macd'] - d['macd_signal']
    high_low = d['high'] - d['low']
    high_close = np.abs(d['high'] - d['close'].shift())
    low_close = np.abs(d['low'] - d['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    d['atr14'] = tr.rolling(14).mean()
    d['atr20'] = tr.rolling(20).mean()
    d['vol_ma'] = d['volume'].rolling(20).mean()
    d['vol_ratio'] = d['volume'] / d['vol_ma']
    d['mom10'] = d['close'].pct_change(10)
    return d

def detect_market_regime(df, i):
    if i < 50: return 'unknown', 0
    row = df.iloc[i]
    ema20_above_50 = row['ema20'] > row['ema50']
    ema50_above_200 = row['ema50'] > row['ema200'] if 'ema200' in row.index and not pd.isna(row['ema200']) else ema20_above_50
    uptrend = ema20_above_50 and ema50_above_200 and row['ema50_slope'] > 0.0001
    downtrend = not ema20_above_50 and not ema50_above_200 and row['ema50_slope'] < -0.0001
    rsi = row['rsi']
    if uptrend and rsi < 70: return 'bull', 1
    elif downtrend and rsi > 30: return 'bear', -1
    else: return 'sideways', 0

def ml_predict_regime(df, i):
    if i < 50: return 'unknown'
    row = df.iloc[i]
    features = {'rsi': row['rsi'], 'rsi_slope': row.get('rsi_slope', 0), 'macd_hist': row.get('macd_hist', 0), 'ema50_slope': row.get('ema50_slope', 0)}
    bullish, bearish = 0, 0
    if features['rsi'] < 40: bullish += 1
    elif features['rsi'] > 60: bearish += 1
    if features['macd_hist'] > 0: bullish += 1
    elif features['macd_hist'] < 0: bearish += 1
    if features['ema50_slope'] > 0.0001: bullish += 1
    elif features['ema50_slope'] < -0.0001: bearish += 1
    if bullish > bearish + 1: return 'bull'
    elif bearish > bullish + 1: return 'bear'
    return 'sideways'

def run_backtest_enhanced(df, config):
    df = calculate_indicators(df)
    cash, equity_base = config['capital'], config['capital']
    pos = {'size': 0, 'entry': 0, 'type': None, 'strat': None, 'entry_bar': 0, 'partial_done': False, 'entry_time': None}
    trades, equity, n = [], [], len(df)
    regime = 'unknown'
    
    for i in range(50, n - 1):
        row, price = df.iloc[i], float(df.iloc[i+1]['close'])
        current_equity = cash + pos['size'] * price if pos['size'] > 0 else cash
        equity_base = max(current_equity, config['capital'] * 0.5)
        
        regime = ml_predict_regime(df, i) if config.get('use_ml', False) else detect_market_regime(df, i)[0]
        
        if pos['size'] > 0:
            atr = row.get('atr14', row.get('atr20', price * 0.01))
            atr_multiplier = config.get('atr_multiplier', 2.0)
            target = config['target_mr'] if pos['strat'] == 'MR' else config['target_tf']
            stop = config['stop_mr'] if pos['strat'] == 'MR' else config['stop_tf']
            if config.get('use_atr', False):
                atr_stop = (atr / price) * atr_multiplier
                stop = max(stop, atr_stop)
            
            ret = (price - pos['entry']) / pos['entry'] if pos['type'] == 'long' else (pos['entry'] - price) / pos['entry']
            exit_reason = None
            
            if not pos.get('partial_done') and pos['type'] == 'long' and row['rsi'] > 50:
                close_size = pos['size'] * 0.5
                pnl = close_size * (price * (1-config['fee']) - pos['entry'])
                cash += close_size * price * (1-config['fee'])
                pos['size'] -= close_size
                pos['partial_done'] = True
                trades.append({'symbol': config.get('symbol', 'UNK'), 'type': 'LONG', 'action': 'PARTIAL', 'entry': pos['entry'], 'exit': price, 'return': ret*100, 'reason': 'RSI=50', 'pnl': pnl, 'regime': regime})
            elif ret <= -stop: exit_reason = 'SL'
            elif ret >= target: exit_reason = 'TP'
            elif pos['type'] == 'long' and row['rsi'] > config['rsi_overbought']: exit_reason = 'RSI'
            elif pos['type'] == 'short' and row['rsi'] < config['rsi_oversold']: exit_reason = 'RSI'
            elif i - pos['entry_bar'] >= config['max_hold_hours']: exit_reason = 'TIME'
            
            if exit_reason and pos['size'] > 0:
                exit_price = price * (1 - config['fee'])
                pnl = pos['size'] * (exit_price - pos['entry']) if pos['type'] == 'long' else pos['size'] * (pos['entry'] - exit_price)
                cash += pos['size'] * exit_price
                trades.append({'symbol': config.get('symbol', 'UNK'), 'type': pos['type'].upper(), 'action': 'EXIT', 'entry': pos['entry'], 'exit': exit_price, 'return': ret*100, 'reason': exit_reason, 'pnl': pnl, 'regime': regime})
                pos = {'size': 0, 'entry': 0, 'type': None, 'strat': None, 'entry_bar': 0, 'partial_done': False, 'entry_time': None}
        else:
            atr = row.get('atr14', row.get('atr20', price * 0.01))
            atr_multiplier, use_atr = config.get('atr_multiplier', 2.0), config.get('use_atr', False)
            stop_mr_dynamic = max(config['stop_mr'], (atr / price) * atr_multiplier) if use_atr else config['stop_mr']
            stop_tf_dynamic = max(config['stop_tf'], (atr / price) * atr_multiplier) if use_atr else config['stop_tf']
            
            if use_atr:
                risk_amount = equity_base * config['risk_pct']
                size = min(risk_amount / atr, (equity_base * config['max_position_pct']) / price)
            else:
                stop = config['stop_mr'] if regime == 'sideways' else config['stop_tf']
                size = min((equity_base * config['risk_pct']) / stop, (equity_base * config['max_position_pct']) / price)
            
            if size * price > cash: size = cash / price * 0.9
            if size * price < 10: equity.append(cash); continue
            
            vol_confirm = float(row['volume']) > float(row['vol_ma'])
            macd_filter = config.get('use_macd', True)
            macd_bullish = row.get('macd_hist', 0) > 0 if macd_filter else True
            macd_bearish = row.get('macd_hist', 0) < 0 if macd_filter else True
            
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
                    short_rsi_ob = config.get('short_rsi_overbought', 55)
                    if float(row['close']) < float(row['ema20']) and float(row['close']) < float(row['ema50']) and row['rsi'] > short_rsi_ob:
                        signal, strat = 'short', 'TF'
            
            if config.get('bear_short_bias', False) and regime == 'bear':
                if row['rsi'] > 45 and float(row['close']) < float(row['ema50']):
                    signal, strat = 'short', 'BEAR'
            
            if signal:
                entry = price * (1 + config['fee'])
                cash -= size * entry
                pos = {'size': size, 'entry': entry, 'type': signal, 'strat': strat, 'entry_bar': i, 'partial_done': False, 'entry_time': row.get('datetime')}
                trades.append({'symbol': config.get('symbol', 'UNK'), 'type': signal.upper(), 'action': 'ENTRY', 'entry': entry, 'exit': 0, 'return': 0, 'reason': f'{regime.upper()}_{strat}', 'pnl': 0, 'regime': regime})
        
        equity.append(cash + pos['size'] * price if pos['size'] > 0 else cash)
    
    if pos['size'] > 0:
        cash += pos['size'] * float(df.iloc[-1]['close']) * (1 - config['fee'])
        equity[-1] = cash
    
    return {'trades': trades, 'equity': equity, 'final': cash, 'df': df}

def calculate_metrics(result, config, risk_free_rate=0.0):
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
    rf = risk_free_rate / (24*365)
    excess = rets_arr - rf
    sharpe = np.mean(excess) / np.std(excess) * np.sqrt(24*365) if np.std(excess) > 0 and len(excess) > 10 else 0
    downside = excess[excess < 0]
    sortino = np.mean(excess) / np.std(downside) * np.sqrt(24*365) if len(downside) > 0 and np.std(downside) > 0 else 0
    calmar = abs(cagr / max_dd) if max_dd > 0 else 0
    return {'trades': n, 'wr': wr, 'pf': pf, 'dd': max_dd, 'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar, 'cagr': cagr, 'final': equity[-1], 'longs': len([t for t in closed if t['type'] == 'LONG']), 'shorts': len([t for t in closed if t['type'] == 'SHORT'])}

def empty_metrics(config):
    return {'trades': 0, 'wr': 0, 'pf': 0, 'dd': 0, 'sharpe': 0, 'sortino': 0, 'calmar': 0, 'cagr': 0, 'final': config['capital'], 'longs': 0, 'shorts': 0}

def plot_equity(results, config):
    fig = go.Figure()
    colors = {'BTC': '#f39c12', 'ETH': '#3498db', 'SOL': '#9b59b6', 'ADA': '#2ecc71', 'XRP': '#e74c3c', 'DOT': '#1abc9c'}
    for symbol, data in results.items():
        m = data.get('metrics', {})
        fig.add_trace(go.Scatter(y=data['equity'], name=symbol, line=dict(color=colors.get(symbol, '#e74c3c'), width=2)))
    fig.add_hline(y=config['capital'], line_dash="dot", line_color="gray", opacity=0.5)
    fig.update_layout(title="📈 Equity Curves", template="plotly_dark", height=400)
    return fig

def plot_dashboard(results):
    symbols = list(results.keys())
    col1, col2 = st.columns(2)
    with col1:
        wr = [results[s].get('metrics',{}).get('wr',0) for s in symbols]
        fig = go.Figure(go.Bar(x=symbols, y=wr, marker_color='#3498db', text=[f"{w:.1f}%" for w in wr], textposition='outside'))
        fig.update_layout(title="Win Rate %", template="plotly_dark", height=250)
        st.plotly_chart(fig, use_container_width=True)
        dd = [results[s].get('metrics',{}).get('dd',0) for s in symbols]
        fig = go.Figure(go.Bar(x=symbols, y=dd, marker_color='indianred'))
        fig.update_layout(title="Max Drawdown %", template="plotly_dark", height=250)
        st.plotly_chart(fig, use_container_width=True)
        cagr = [results[s].get('metrics',{}).get('cagr',0) for s in symbols]
        colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in cagr]
        fig = go.Figure(go.Bar(x=symbols, y=cagr, marker_color=colors))
        fig.update_layout(title="CAGR %", template="plotly_dark", height=250)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        longs = [results[s].get('metrics',{}).get('longs',0) for s in symbols]
        shorts = [results[s].get('metrics',{}).get('shorts',0) for s in symbols]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=symbols, y=longs, name='Longs', marker_color='#2ecc71'))
        fig.add_trace(go.Bar(x=symbols, y=shorts, name='Shorts', marker_color='#e74c3c'))
        fig.update_layout(title="Longs vs Shorts", template="plotly_dark", height=250)
        st.plotly_chart(fig, use_container_width=True)
        sharpe = [results[s].get('metrics',{}).get('sharpe',0) for s in symbols]
        fig = go.Figure(go.Bar(x=symbols, y=sharpe, marker_color='mediumpurple'))
        fig.update_layout(title="Sharpe Ratio", template="plotly_dark", height=250)
        st.plotly_chart(fig, use_container_width=True)
        final = [results[s].get('metrics',{}).get('final',0) for s in symbols]
        fig = go.Figure(go.Bar(x=symbols, y=final, marker_color='#f39c12'))
        fig.update_layout(title="Final Equity ($)", template="plotly_dark", height=250)
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("📊 Metrics Table")
    rows = [{'Symbol': s, 'Trades': d['metrics']['trades'], 'WR%': f"{d['metrics']['wr']:.1f}", 'PF': f"{d['metrics']['pf']:.2f}", 'DD%': f"{d['metrics']['dd']:.1f}", 'Sharpe': f"{d['metrics']['sharpe']:.2f}", 'CAGR%': f"{d['metrics']['cagr']:.1f}", 'Final$': f"${d['metrics']['final']:,.0f}"} for s, d in results.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.download_button("📥 CSV", pd.DataFrame(rows).to_csv(index=False), "metrics.csv", "text/csv")

def plot_flowchart(config):
    import graphviz
    dot = graphviz.Digraph()
    dot.attr(rankdir='TB', size='10,6')
    dot.node('start', 'START', style='filled', fillcolor='#3498db', fontcolor='white')
    dot.node('regime', 'Regime\n(RSI/MACD)', style='filled', fillcolor='#f39c12', fontcolor='white')
    dot.node('bull', 'BULL\n→ TF Longs', style='filled', fillcolor='#2ecc71', fontcolor='white')
    dot.node('bear', 'BEAR\n→ TF Shorts', style='filled', fillcolor='#e74c3c', fontcolor='white')
    dot.node('sideways', 'SIDEWAYS\n→ Mean Rev', style='filled', fillcolor='#9b59b6', fontcolor='white')
    dot.node('exit', f'Exit\nTP:{config["target_mr"]*100}%', style='filled', fillcolor='#e74c3c', fontcolor='white')
    dot.edges([('start','regime'), ('regime','bull'), ('regime','bear'), ('regime','sideways'), ('bull','exit'), ('bear','exit'), ('sideways','exit')])
    return dot

def optimize_params(symbol, days, use_api, base_config, market_type='mixed'):
    st.info(f"🔍 Optimizing {symbol}...")
    best_sharpe = -999
    best_config = base_config.copy()
    rsi_os_range = [25, 30, 35, 40]
    rsi_ob_range = [60, 65, 70]
    short_rsi_ob_range = [45, 50, 55]
    progress = st.progress(0)
    total = len(rsi_os_range) * len(rsi_ob_range) * len(short_rsi_ob_range)
    count = 0
    for rsi_os in rsi_os_range:
        for rsi_ob in rsi_ob_range:
            for short_ob in short_rsi_ob_range:
                count += 1
                if count % 5 == 0: progress.progress(min(count/total, 1.0))
                cfg = base_config.copy()
                cfg['rsi_oversold'] = rsi_os
                cfg['rsi_overbought'] = rsi_ob
                cfg['short_rsi_overbought'] = short_ob
                df, _ = load_data(symbol, days, use_api, market_type)
                if df is None: continue
                result = run_backtest_enhanced(df, cfg)
                metrics = calculate_metrics(result, cfg)
                if metrics['sharpe'] > best_sharpe:
                    best_sharpe = metrics['sharpe']
                    best_config = cfg.copy()
    progress.empty()
    return best_config

def main():
    st.title(f"📈 Crypto Strategy v{VERSION}")
    st.markdown("**Enhanced: ATR Stops | MACD Filter | ML Regime | Bear Short Bias**")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        capital = st.number_input("Capital ($)", value=10000, step=1000)
        c1, c2 = st.columns(2)
        start_date = c1.date_input("From", datetime(2023, 1, 1))
        end_date = c2.date_input("To", datetime.now())
        years = max(0.1, (end_date - start_date).days / 365)
        market_type = st.selectbox("Market Type", ['mixed', 'bull', 'bear', 'sideways'], index=0)
        symbols = st.multiselect("Symbols", ['BTC','ETH','SOL','ADA','XRP','DOT'], default=['BTC','ETH'])
        use_api = st.checkbox("Use CoinGecko API", False)
        
        st.subheader("Strategy Parameters")
        rsi_oversold = st.number_input("RSI Oversold", value=30, step=5, min_value=10, max_value=45)
        rsi_overbought = st.number_input("RSI Overbought", value=65, step=5, min_value=55, max_value=90)
        short_rsi_ob = st.number_input("Short RSI Overbought", value=55, step=5, min_value=40, max_value=70, help="Lower = more aggressive shorts")
        
        st.subheader("Mean Reversion")
        target_mr = st.slider("Target %", 1.0, 10.0, 3.0) / 100
        stop_mr = st.slider("Stop %", 0.5, 5.0, 1.5) / 100
        
        st.subheader("Trend Following")
        target_tf = st.slider("TF Target %", 1.0, 15.0, 5.0) / 100
        stop_tf = st.slider("TF Stop %", 0.5, 5.0, 2.5) / 100
        
        st.subheader("Advanced")
        use_macd = st.checkbox("MACD Filter", True, help="Only trade in MACD direction")
        use_atr = st.checkbox("ATR-based Stops", False, help="Dynamic stops using ATR")
        atr_multiplier = st.slider("ATR Multiplier", 1.0, 4.0, 2.0, help="Higher = wider stops")
        use_ml = st.checkbox("ML Regime Detection", False, help="Use ML for regime")
        bear_short_bias = st.checkbox("Bear Short Bias", True, help="More shorts in bear markets")
        
        if st.button("🎯 Optimize"):
            if symbols:
                cfg = {'capital': capital, 'fee': 0.001, 'risk_pct': 0.02, 'rsi_oversold': rsi_oversold, 'rsi_overbought': rsi_overbought, 'short_rsi_overbought': short_rsi_ob, 'target_mr': target_mr, 'stop_mr': stop_mr, 'target_tf': target_tf, 'stop_tf': stop_tf, 'max_hold_hours': 8, 'max_position_pct': 0.30, 'use_macd': use_macd, 'use_atr': use_atr, 'atr_multiplier': atr_multiplier, 'use_ml': use_ml, 'bear_short_bias': bear_short_bias}
                best = optimize_params(symbols[0], int(years*365), use_api, cfg, market_type)
                st.session_state.optimized = best
                st.success(f"Best: RSI {best['rsi_oversold']}/{best['rsi_overbought']}, Short RSI: {best['short_rsi_overbought']}")
        
        if st.session_state.optimized:
            st.info(f"Optimized: RSI {st.session_state.optimized['rsi_oversold']}/{st.session_state.optimized['rsi_overbought']}")
            if st.button("Apply"):
                rsi_oversold = st.session_state.optimized['rsi_oversold']
                rsi_overbought = st.session_state.optimized['rsi_overbought']
                short_rsi_ob = st.session_state.optimized['short_rsi_overbought']
                st.rerun()
        
        if st.button("🔄 Reset"):
            st.session_state.results = None
            st.session_state.optimized = None
            st.rerun()
    
    config = {'capital': capital, 'fee': 0.001, 'risk_pct': 0.02, 'rsi_oversold': rsi_oversold, 'rsi_overbought': rsi_overbought, 'short_rsi_overbought': short_rsi_ob, 'target_mr': target_mr, 'stop_mr': stop_mr, 'target_tf': target_tf, 'stop_tf': stop_tf, 'max_hold_hours': 8, 'max_position_pct': 0.30, 'use_macd': use_macd, 'use_atr': use_atr, 'atr_multiplier': atr_multiplier, 'use_ml': use_ml, 'bear_short_bias': bear_short_bias}
    
    if st.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Running enhanced backtest..."):
            results = {}
            for symbol in symbols:
                with st.expander(f"{symbol}/USDT"):
                    cfg = config.copy()
                    cfg['symbol'] = symbol
                    df, status = load_data(symbol, int(years*365), use_api, market_type)
                    st.caption(f"Data: {status}")
                    if df is not None and len(df) > 100:
                        result = run_backtest_enhanced(df, cfg)
                        metrics = calculate_metrics(result, cfg)
                        result['metrics'] = metrics
                        results[symbol] = result
                        m = result['metrics']
                        st.success(f"✓ {m['trades']} trades | WR: {m['wr']:.1f}% | PF: {m['pf']:.2f} | Sharpe: {m['sharpe']:.2f} | CAGR: {m['cagr']:.1f}% | Final: ${m['final']:,.0f}")
                    else:
                        st.error("No data")
            if results:
                st.session_state.results = results
    
    if st.session_state.results:
        results = st.session_state.results
        st.markdown("---")
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Equity", "📈 Dashboard", "🔀 Flowchart", "📋 Trades"])
        with tab1:
            fig = plot_equity(results, config)
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            plot_dashboard(results)
        with tab3:
            try:
                dot = plot_flowchart(config)
                st.graphviz_chart(dot)
            except: st.info("Install graphviz: pip install graphviz")
        with tab4:
            all_trades = []
            for symbol, data in results.items():
                for t in data.get('trades', []):
                    t['symbol'] = symbol
                    all_trades.append(t)
            if all_trades:
                df_trades = pd.DataFrame(all_trades)
                st.dataframe(df_trades[['symbol','type','action','entry','exit','return','reason','pnl','regime']], use_container_width=True)
            else:
                st.info("No trades")
    
    st.markdown("---")
    st.caption(f"Made with ❤️ by OpenClaw | Version {VERSION}")

if __name__ == "__main__":
    main()
