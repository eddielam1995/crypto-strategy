#!/usr/bin/env python3
"""
Quantitative Crypto Trading Strategy - Streamlit App v3
======================================================
Enhanced with:
- Rate limiting & retry for CoinGecko
- Data-driven order book simulation
- Proper partial exit logging
- Date range picker
- Parameter optimization grid search
- Sortino with risk-free rate
- Mobile responsive

Version: 3.0
DEPLOY: streamlit run crypto_streamlit_app_v3.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
from io import BytesIO

# Page config
st.set_page_config(
    page_title="Crypto Strategy v3",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

VERSION = "3.0"

# Session State
if 'results' not in st.session_state:
    st.session_state.results = None
if 'config' not in st.session_state:
    st.session_state.config = None
if 'optimized' not in st.session_state:
    st.session_state.optimized = None

COIN_MAP = {
    'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana',
    'ADA': 'cardano', 'XRP': 'ripple', 'DOT': 'polkadot',
    'DOGE': 'dogecoin', 'AVAX': 'avalanche-2',
}

# ===============================
# DATA FUNCTIONS
# ===============================

@st.cache_data(ttl=3600)
def fetch_coingecko(coin_id, days, _retry_count=3):
    """Fetch with exponential backoff retry"""
    from pycoingecko import CoinGeckoAPI
    cg = CoinGeckoAPI()
    
    for attempt in range(_retry_count):
        try:
            try:
                data = cg.get_coin_market_chart_by_id(coin_id, 'usd', days, interval='hourly')
                if data and len(data.get('prices', [])) > 100:
                    prices = data['prices']
                    df = pd.DataFrame(prices, columns=['ts', 'close'])
                    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
                    df = df.set_index('datetime')
                    df = df.resample('1h').agg({'close': 'last'}).dropna().reset_index()
                    return df, 'hourly'
            except:
                pass
            
            data = cg.get_coin_market_chart_by_id(coin_id, 'usd', days)
            prices = data['prices']
            df = pd.DataFrame(prices, columns=['ts', 'close'])
            df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
            df = df.set_index('datetime')
            df = df.resample('1h').interpolate(method='linear').reset_index()
            return df, 'daily-upsampled'
        except Exception as e:
            if attempt < _retry_count - 1:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                return None, f"Error: {str(e)[:50]}"
    return None, "Max retries"

def generate_demo_data(symbol, days):
    np.random.seed(42 if symbol == 'BTC' else hash(symbol) % 1000)
    hours = days * 24
    base = {'BTC': 45000, 'ETH': 2500, 'SOL': 100, 'ADA': 1, 
            'XRP': 0.5, 'DOT': 7, 'DOGE': 0.08, 'AVAX': 35}.get(symbol, 1000)
    
    n = hours
    bull, bear = int(n*0.25), int(n*0.25)
    returns = np.zeros(n)
    returns[:bull] = np.random.normal(0.0003, 0.03/np.sqrt(24), bull)
    returns[bull:bull+bear] = np.random.normal(-0.0002, 0.03/np.sqrt(24), bear)
    returns[bull+bear:] = np.random.normal(0.00005, 0.02/np.sqrt(24), n-bull-bear)
    prices = base * np.exp(np.cumsum(returns))
    
    # Volatility-based order book
    vol = pd.Series(returns).rolling(20).std().fillna(0.02).values
    bid_ask_ratio = np.clip(1 + vol * np.random.uniform(-2, 2, n), 0.5, 2.0)
    
    return pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=n, freq='h'),
        'open': prices * np.random.uniform(0.99, 1.01, n),
        'high': prices * np.random.uniform(1.00, 1.02, n),
        'low': prices * np.random.uniform(0.98, 1.00, n),
        'close': prices,
        'volume': np.random.uniform(100, 5000, n),
        'bid_ask_ratio': bid_ask_ratio
    })

@st.cache_data
def load_data(symbol, days, use_api):
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
    return generate_demo_data(symbol, days), 'demo'

# ===============================
# INDICATORS
# ===============================

def calculate_indicators(df):
    d = df.copy()
    delta = d['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean().fillna(0)
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().fillna(0).replace(0, 0.0001)
    d['rsi'] = (100 - (100 / (1 + gain / loss))).clip(0, 100)
    d['bb_mid'] = d['close'].rolling(20).mean()
    d['bb_std'] = d['close'].rolling(20).std().fillna(0)
    d['bb_lower'] = d['bb_mid'] - 2 * d['bb_std']
    d['bb_upper'] = d['bb_mid'] + 2 * d['bb_std']
    d['ema20'] = d['close'].ewm(span=20, adjust=False).mean()
    d['ema50'] = d['close'].ewm(span=50, adjust=False).mean()
    d['vol_ma'] = d['volume'].rolling(20).mean()
    return d

def detect_regime(df, i):
    if i < 20: return 'trending'
    rsi = df.iloc[i]['rsi']
    return 'ranging' if 40 <= rsi <= 60 else 'trending'

# ===============================
# BACKTEST ENGINE
# ===============================

def run_backtest(df, config):
    df = calculate_indicators(df)
    cash = config['capital']
    equity_base = config['capital']
    pos = {'size': 0, 'entry': 0, 'type': None, 'strat': None, 'entry_bar': 0, 
           'partial_done': False, 'entry_time': None}
    trades, equity = [], []
    n = len(df)
    
    for i in range(50, n - 1):
        row = df.iloc[i]
        price = float(df.iloc[i+1]['close'])
        current_equity = cash + pos['size'] * price if pos['size'] > 0 else cash
        equity_base = max(current_equity, config['capital'] * 0.5)
        
        if pos['size'] > 0:
            target = config['target_mr'] if pos['strat'] == 'MR' else config['target_tf']
            stop = config['stop_mr'] if pos['strat'] == 'MR' else config['stop_tf']
            ret = (price - pos['entry']) / pos['entry'] if pos['type'] == 'long' else (pos['entry'] - price) / pos['entry']
            exit_reason = None
            
            # Partial exit at RSI 50
            if not pos.get('partial_done') and pos['type'] == 'long' and row['rsi'] > 50:
                close_size = pos['size'] * 0.5
                pnl = close_size * (price * (1-config['fee']) - pos['entry'])
                cash += close_size * price * (1-config['fee'])
                pos['size'] -= close_size
                pos['partial_done'] = True
                trades.append({
                    'symbol': config.get('symbol', 'UNK'), 'type': 'LONG', 'action': 'PARTIAL EXIT',
                    'entry': pos['entry'], 'exit': price, 'return': ret*100, 'reason': 'RSI=50',
                    'pnl': pnl, 'size': close_size
                })
            elif ret <= -stop: exit_reason = 'SL'
            elif ret >= target: exit_reason = 'TP'
            elif pos['type'] == 'long' and row['rsi'] > config['rsi_overbought']: exit_reason = 'RSI'
            elif pos['type'] == 'short' and row['rsi'] < config['rsi_oversold']: exit_reason = 'RSI'
            elif i - pos['entry_bar'] >= config['max_hold_hours']: exit_reason = 'TIME'
            
            if exit_reason and pos['size'] > 0:
                exit_price = price * (1 - config['fee'])
                pnl = pos['size'] * (exit_price - pos['entry']) if pos['type'] == 'long' else pos['size'] * (pos['entry'] - exit_price)
                cash += pos['size'] * exit_price
                trades.append({
                    'symbol': config.get('symbol', 'UNK'), 'type': pos['type'].upper(), 'action': 'EXIT',
                    'entry': pos['entry'], 'exit': exit_price, 'return': ret*100, 
                    'reason': exit_reason, 'pnl': pnl, 'size': pos['size']
                })
                pos = {'size': 0, 'entry': 0, 'type': None, 'strat': None, 
                       'entry_bar': 0, 'partial_done': False, 'entry_time': None}
        else:
            market = detect_regime(df, i)
            stop = config['stop_mr'] if market == 'ranging' else config['stop_tf']
            size = min((equity_base * config['risk_pct']) / stop, 
                      (equity_base * config['max_position_pct']) / price)
            if size * price > cash: size = cash / price * 0.9
            if size * price < 10:
                equity.append(cash)
                continue
            
            vol_confirm = float(row['volume']) > float(row['vol_ma'])
            ob_confirm = row.get('bid_ask_ratio', 1.0) > 1.0 if row.get('bid_ask_ratio') else True
            signal = strat = None
            
            if market == 'ranging':
                if row['rsi'] < config['rsi_oversold'] and float(row['close']) > float(row['ema50']) and vol_confirm:
                    signal, strat = 'long', 'MR'
                elif row['rsi'] > config['rsi_overbought'] and float(row['close']) < float(row['ema50']) and vol_confirm:
                    signal, strat = 'short', 'MR'
            else:
                if float(row['close']) > float(row['ema20']) and float(row['close']) > float(row['ema50']) and row['rsi'] < 60:
                    signal, strat = 'long', 'TF'
                elif float(row['close']) < float(row['ema20']) and float(row['close']) < float(row['ema50']) and row['rsi'] > 40:
                    signal, strat = 'short', 'TF'
            
            if signal:
                entry = price * (1 + config['fee'])
                cash -= size * entry
                pos = {'size': size, 'entry': entry, 'type': signal, 'strat': strat, 
                       'entry_bar': i, 'partial_done': False, 'entry_time': row.get('datetime')}
                trades.append({
                    'symbol': config.get('symbol', 'UNK'), 'type': signal.upper(), 'action': 'ENTRY',
                    'entry': entry, 'exit': 0, 'return': 0, 
                    'reason': f'{market.upper()}_{strat}', 'pnl': 0, 'size': size
                })
        
        equity.append(cash + pos['size'] * price if pos['size'] > 0 else cash)
    
    if pos['size'] > 0:
        cash += pos['size'] * float(df.iloc[-1]['close']) * (1 - config['fee'])
        equity[-1] = cash
    
    return {'trades': trades, 'equity': equity, 'final': cash, 'df': df}

def calculate_metrics(result, config, risk_free_rate=0.0):
    trades, equity = result['trades'], result['equity']
    if not equity:
        return empty_metrics(config)
    
    closed = [t for t in trades if t.get('action') == 'EXIT']
    if not closed:
        return empty_metrics(config)
    
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
    
    return {
        'trades': n, 'wr': wr, 'pf': pf, 'dd': max_dd,
        'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar,
        'cagr': cagr, 'final': equity[-1],
        'longs': len([t for t in closed if t['type'] == 'LONG']),
        'shorts': len([t for t in closed if t['type'] == 'SHORT']),
    }

def empty_metrics(config):
    return {'trades': 0, 'wr': 0, 'pf': 0, 'dd': 0, 'sharpe': 0, 
            'sortino': 0, 'calmar': 0, 'cagr': 0, 'final': config['capital'], 'longs': 0, 'shorts': 0}

# ===============================
# OPTIMIZATION
# ===============================

def optimize_params(symbol, days, use_api, base_config):
    st.info(f"🔍 Optimizing {symbol}...")
    best_sharpe = -999
    best_config = base_config.copy()
    
    rsi_oversold_range = [30, 35, 40]
    rsi_overbought_range = [60, 65, 70]
    target_mr_range = [0.02, 0.025, 0.03]
    
    progress = st.progress(0)
    total = len(rsi_oversold_range) * len(rsi_overbought_range) * len(target_mr_range)
    count = 0
    
    for rsi_os in rsi_oversold_range:
        for rsi_ob in rsi_overbought_range:
            for tgt in target_mr_range:
                count += 1
                progress.progress(count / total)
                cfg = base_config.copy()
                cfg['rsi_oversold'] = rsi_os
                cfg['rsi_overbought'] = rsi_ob
                cfg['target_mr'] = tgt
                
                df, _ = load_data(symbol, days, use_api)
                if df is None:
                    continue
                result = run_backtest(df, cfg)
                metrics = calculate_metrics(result, cfg)
                if metrics['sharpe'] > best_sharpe:
                    best_sharpe = metrics['sharpe']
                    best_config = cfg.copy()
                    best_config['sharpe'] = best_sharpe
    
    progress.empty()
    return best_config

# ===============================
# VISUALIZATIONS
# ===============================

def plot_equity(results, config, show_bh=False):
    n = len(results)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=list(results.keys()), vertical_spacing=0.08)
    colors = {'BTC': '#f39c12', 'ETH': '#3498db', 'SOL': '#9b59b6', 'ADA': '#2ecc71', 'XRP': '#e74c3c', 'DOT': '#1abc9c'}
    
    for idx, (symbol, data) in enumerate(results.items()):
        row, col = idx // cols + 1, idx % cols + 1
        m = data.get('metrics', {})
        fig.add_trace(go.Scatter(y=data['equity'], name=symbol, line=dict(color=colors.get(symbol, '#e74c3c'), width=1.5)), row=row, col=col)
        if show_bh and 'df' in data:
            bh = config['capital'] * data['df']['close'] / data['df']['close'].iloc[0]
            fig.add_trace(go.Scatter(y=bh, name=f'{symbol} BH', line=dict(color='gray', width=1, dash='dash')), row=row, col=col)
        fig.add_hline(y=config['capital'], line_dash="dot", line_color="red", row=row, col=col)
        fig.add_annotation(x=0.02, y=0.98, xref=f'x{idx+1}', yref=f'y{idx+1}', text=f"WR: {m.get('wr',0):.1f}% | PF: {m.get('pf',0):.2f}", showarrow=False, font=dict(size=9), bgcolor='white', xanchor='left', yanchor='top', row=row, col=col)
    
    fig.update_layout(height=300*rows, title_text="Equity Curves", showlegend=False, template="plotly_dark")
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig

def plot_dashboard(results):
    symbols = list(results.keys())
    fig = make_subplots(rows=2, cols=3, subplot_titles=('Win Rate %', 'Max DD %', 'Sharpe', 'Direction', 'Table', ''))
    
    wr = [results[s].get('metrics',{}).get('wr',0) for s in symbols]
    fig.add_trace(go.Bar(x=symbols, y=wr, marker_color=['#3498db','#2ecc71','#e74c3c','#9b59b6'][:len(symbols)]), row=1, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="red", row=1, col=1)
    
    dd = [results[s].get('metrics',{}).get('dd',0) for s in symbols]
    fig.add_trace(go.Bar(x=symbols, y=dd, marker_color='indianred'), row=1, col=2)
    
    sharpe = [results[s].get('metrics',{}).get('sharpe',0) for s in symbols]
    fig.add_trace(go.Bar(x=symbols, y=sharpe, marker_color='mediumpurple'), row=1, col=3)
    
    longs = sum(results[s].get('metrics',{}).get('longs',0) for s in symbols)
    shorts = sum(results[s].get('metrics',{}).get('shorts',0) for s in symbols)
    fig.add_trace(go.Pie(labels=['Longs', 'Shorts'], values=[longs, shorts], marker_colors=['#2ecc71', '#e74c3c'], hole=0.4), row=2, col=1)
    
    table_data = [[s, results[s].get('metrics',{}).get('trades',0), f"{results[s].get('metrics',{}).get('wr',0):.1f}%", f"{results[s].get('metrics',{}).get('sharpe',0):.2f}", f"${results[s].get('metrics',{}).get('final',0):,.0f}"] for s in symbols]
    fig.add_trace(go.Table(header=dict(values=['Symbol','Trades','WR%','Sharpe','Final$'], fill_color='royalblue'), cells=dict(values=list(zip(*table_data)), fill_color='lavender')), row=2, col=2)
    
    fig.update_layout(height=600, title_text="Dashboard", template="plotly_dark")
    return fig

def plot_flowchart(config):
    import graphviz
    dot = graphviz.Digraph()
    dot.attr(rankdir='TB', size='10,6')
    dot.node('start', 'START', style='filled', fillcolor='#3498db', fontcolor='white')
    dot.node('regime', f'RSI Check\n({config["rsi_oversold"]}-{config["rsi_overbought"]})', style='filled', fillcolor='#f39c12', fontcolor='white')
    dot.node('ranging', 'RANGING\n→ Mean Reversion', style='filled', fillcolor='#2ecc71', fontcolor='white')
    dot.node('trend', 'TRENDING\n→ Trend Following', style='filled', fillcolor='#9b59b6', fontcolor='white')
    dot.node('mr', f'MR\nT:{config["target_mr"]*100}% S:{config["stop_mr"]*100}%', style='filled', fillcolor='#27ae60', fontcolor='white')
    dot.node('tf', f'TF\nT:{config["target_tf"]*100}% S:{config["stop_tf"]*100}%', style='filled', fillcolor='#8e44ad', fontcolor='white')
    dot.node('exit', 'EXIT\nTP|SL|RSI|TIME', style='filled', fillcolor='#e74c3c', fontcolor='white')
    dot.edges([('start','regime'), ('regime','ranging'), ('regime','trend'), ('ranging','mr'), ('trend','tf'), ('mr','exit'), ('tf','exit')])
    return dot

# ===============================
# MAIN
# ===============================

def main():
    st.title(f"📈 Crypto Strategy v{VERSION}")
    st.markdown("**Dual Regime: Mean Reversion + Trend Following**")
    
    with st.sidebar:
        st.header("⚙️ Config")
        capital = st.number_input("Capital ($)", value=10000, step=1000)
        
        st.subheader("Date Range")
        c1, c2 = st.columns(2)
        start_date = c1.date_input("From", datetime(2023, 1, 1))
        end_date = c2.date_input("To", datetime.now())
        years = max(0.1, (end_date - start_date).days / 365)
        
        symbols = st.multiselect("Symbols", ['BTC','ETH','SOL','ADA','XRP','DOT','DOGE','AVAX'], default=['BTC','ETH'])
        if len(symbols) > 4:
            st.warning("⚠️ 4+ symbols = slower")
        
        use_api = st.checkbox("Use CoinGecko API", False)
        if use_api:
            st.caption("⚠️ Rate limited ~50/min")
        
        rsi_oversold = st.number_input("RSI Oversold", value=35, step=5, min_value=10, max_value=45)
        rsi_overbought = st.number_input("RSI Overbought", value=65, step=5, min_value=55, max_value=90)
        
        st.subheader("Mean Reversion")
        target_mr = st.slider("Target %", 1.0, 10.0, 2.5) / 100
        stop_mr = st.slider("Stop %", 0.5, 5.0, 1.5) / 100
        
        st.subheader("Trend Following")
        target_tf = st.slider("TF Target %", 1.0, 15.0, 4.0) / 100
        stop_tf = st.slider("TF Stop %", 0.5, 5.0, 2.0) / 100
        
        with st.expander("Advanced"):
            risk_free = st.number_input("Risk-Free Rate %", value=0.0, step=0.5) / 100
            show_bh = st.checkbox("Show Buy & Hold", False)
        
        if st.button("🎯 Optimize"):
            if symbols:
                cfg = {'capital': capital, 'fee': 0.001, 'risk_pct': 0.02, 'rsi_oversold': rsi_oversold, 
                       'rsi_overbought': rsi_overbought, 'target_mr': target_mr, 'stop_mr': stop_mr,
                       'target_tf': target_tf, 'stop_tf': stop_tf, 'max_hold_hours': 8, 'max_position_pct': 0.30}
                best = optimize_params(symbols[0], int(years*365), use_api, cfg)
                st.session_state.optimized = best
                st.success(f"Best: RSI {best['rsi_oversold']}/{best['rsi_overbought']}, Target {best['target_mr']*100}%")
        
        if st.session_state.optimized:
            st.info(f"Using optimized: RSI {st.session_state.optimized['rsi_oversold']}/{st.session_state.optimized['rsi_overbought']}")
            if st.button("Apply"):
                rsi_oversold = st.session_state.optimized['rsi_oversold']
                rsi_overbought = st.session_state.optimized['rsi_overbought']
                target_mr = st.session_state.optimized['target_mr']
                st.rerun()
        
        if st.button("🔄 Reset"):
            st.session_state.results = None
            st.session_state.optimized = None
            st.rerun()
    
    config = {'capital': capital, 'fee': 0.001, 'risk_pct': 0.02, 'rsi_oversold': rsi_oversold, 
              'rsi_overbought': rsi_overbought, 'target_mr': target_mr, 'stop_mr': stop_mr,
              'target_tf': target_tf, 'stop_tf': stop_tf, 'max_hold_hours': 8, 'max_position_pct': 0.30}
    
    if st.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Running..."):
            results = {}
            for symbol in symbols:
                with st.expander(f"{symbol}/USDT"):
                    cfg = config.copy()
                    cfg['symbol'] = symbol
                    df, status = load_data(symbol, int(years*365), use_api)
                    st.caption(f"Data: {status}")
                    if df is not None and len(df) > 100:
                        result = run_backtest(df, cfg)
                        metrics = calculate_metrics(result, cfg, risk_free)
                        result['metrics'] = metrics
                        results[symbol] = result
                        m = result['metrics']
                        st.success(f"✓ {m['trades']} trades | WR: {m['wr']:.1f}% | Sharpe: {m['sharpe']:.2f} | Final: ${m['final']:,.0f}")
                    else:
                        st.error("No data")
            
            if results:
                st.session_state.results = results
                st.session_state.config = config
    
    if st.session_state.results:
        results = st.session_state.results
        st.markdown("---")
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Equity", "📈 Dashboard", "🔀 Flowchart", "📋 Trades"])
        
        with tab1:
            fig = plot_equity(results, config, show_bh)
            st.plotly_chart(fig, use_container_width=True)
            buf = BytesIO()
            fig.write_image(buf, format='png', width=1200, height=600)
            st.download_button("📥 Download", buf.getvalue(), "equity.png", "image/png")
        
        with tab2:
            fig = plot_dashboard(results)
            st.plotly_chart(fig, use_container_width=True)
            rows = [{'Symbol': s, 'Trades': d['metrics']['trades'], 'WR%': f"{d['metrics']['wr']:.1f}", 
                     'PF': f"{d['metrics']['pf']:.2f}", 'DD%': f"{d['metrics']['dd']:.1f}",
                     'Sharpe': f"{d['metrics']['sharpe']:.2f}", 'Sortino': f"{d['metrics']['sortino']:.2f}",
                     'Calmar': f"{d['metrics']['calmar']:.2f}", 'Final$': f"${d['metrics']['final']:,.0f}"} 
                    for s, d in results.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.download_button("📥 CSV", pd.DataFrame(rows).to_csv(index=False), "metrics.csv", "text/csv")
        
        with tab3:
            try:
                dot = plot_flowchart(config)
                st.graphviz_chart(dot)
            except:
                st.info("Install graphviz for flowchart: pip install graphviz")
        
        with tab4:
            all_trades = []
            for symbol, data in results.items():
                for t in data.get('trades', []):
                    t['symbol'] = symbol
                    all_trades.append(t)
            if all_trades:
                df_trades = pd.DataFrame(all_trades)
                st.dataframe(df_trades[['symbol','type','action','entry','exit','return','reason','pnl']], use_container_width=True)
            else:
                st.info("No trades")
    
    st.markdown("---")
    st.caption(f"Made with ❤️ by OpenClaw | Version {VERSION} | Deploy: streamlit run crypto_streamlit_app_v3.py")

if __name__ == "__main__":
    main()
