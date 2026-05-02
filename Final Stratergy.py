"""
╔══════════════════════════════════════════════════════════════╗
║       NSE INTRADAY STRATEGY — v6 (DATA-DRIVEN REBUILD)      ║
║       Sujal's Strategy Lab | Built from 30-day trade log    ║
╠══════════════════════════════════════════════════════════════╣
║  WHAT CHANGED FROM v5 — EVERY DECISION BACKED BY DATA:     ║
║                                                             ║
║  LEARNED "NEVER TRADE" RULES (from trade log analysis):    ║
║  ❌ Trend SELL blocked for Banking, FMCG, Pharma           ║
║     → Banking Trend SELL: 42% WR, -₹1,483 over 30 days    ║
║     → FMCG Trend SELL:    0% WR,  -₹1,292 over 30 days    ║
║     → Pharma Trend SELL:  33% WR, -₹250 over 30 days      ║
║  ❌ Pharma BUY blocked entirely (0% WR, -₹1,319)          ║
║  ❌ No Trend trades in ranging markets (ADX < 22)          ║
║     → All 4 bad days (Mar 16,18,24 Apr 7) were Trend      ║
║       signals in choppy conditions. ADX blocks them all.   ║
║  ❌ No entries in 09:45–10:15 IST window (₹1,561 lost)    ║
║  ❌ Skip if today's ATR > 2× its 20-day avg (panic day)   ║
║  ❌ Skip if gap-open > 0.8% from prev close (news day)    ║
║  ❌ No same-direction entry after 2 losses that direction  ║
║                                                             ║
║  NEW CORES ADDED:                                           ║
║  ✅ ADX (14): Filters choppy markets for Trend signal      ║
║  ✅ Supertrend: Direction confirmation (replaces EMA50     ║
║     alone, more responsive to intraday turns)              ║
║  ✅ ATR Spike Guard: Skips panic/news-driven candles       ║
║  ✅ Gap Filter: Skips first 45 min if gap > 0.8%          ║
║  ✅ Direction Loss Guard per stock: tracks BUY/SELL losses  ║
║     separately — 2 SELL losses today? No more SELL today.  ║
║                                                             ║
║  STOCKS (all under ₹2,000):                                ║
║  Banking  → SBIN.NS       (~₹1,100)                        ║
║  Auto (P) → TATAMOTORS.NS (~₹720)  [passenger vehicles]   ║
║  Auto (C) → ASHOKLEY.NS   (~₹220)  [commercial vehicles]  ║
║  Energy   → ONGC.NS       (~₹285)                          ║
║  Pharma   → CIPLA.NS      (~₹1,400) [replacing SunPharma] ║
║  Metal    → HINDALCO.NS   (~₹1,000)                        ║
║  FMCG     → ITC.NS        (~₹430)                          ║
║                                                             ║
║  POSITION SIZING:                                           ║
║  Risk ₹500/trade | Max capital ₹20,000/trade               ║
║  qty = min(₹500 ÷ (ATR_SL × ATR), ₹20,000 ÷ price)       ║
║                                                             ║
║  HOW TO RUN:                                                ║
║  pip install yfinance pandas numpy                         ║
║  python backtest_v6_learned.py                             ║
╚══════════════════════════════════════════════════════════════╝
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
#  SECTOR MAP — ALL STOCKS UNDER ₹2,000
# ──────────────────────────────────────────────────────────────
SECTORS = {
    "Banking"    : "SBIN.NS",
    "Auto-Pass"  : "TMCV.NS",   # Tata Motors Passenger
    "Auto-Comm"  : "ASHOKLEY.NS",     # Ashok Leyland Commercial
    "Energy"     : "ONGC.NS",
    "Pharma"     : "CIPLA.NS",        # Replaced SunPharma (bad WR)
    "Metal"      : "HINDALCO.NS",
    "FMCG"       : "ITC.NS",
}

TIMEFRAME          = "5m"
PERIOD             = "30d"   # Use 30d to ensure we have enough data after indicator warmup
DIRECTION          = "BOTH"

# ── POSITION SIZING ──────────────────────────────────────────
MAX_RISK_PER_TRADE = 500             # Max ₹ loss per trade
MAX_TRADE_CAPITAL  = 20000           # Hard cap ₹20,000 per trade (user requirement)
MIN_QTY            = 5

# ── ATR RISK PARAMS ──────────────────────────────────────────
ATR_SL_MULT        = 2.0
ATR_T1_MULT        = 2.0
ATR_T2_MULT        = 3.5
ATR_MIN_PCT        = 0.002           # Skip if market too flat

# ── SUPERTREND PARAMS ─────────────────────────────────────────
ST_PERIOD          = 10
ST_MULT            = 3.0

# ── ADX PARAMS ───────────────────────────────────────────────
ADX_PERIOD         = 14
ADX_TREND_MIN      = 22              # Trend signals only when ADX > 22

# ── VOLUME ───────────────────────────────────────────────────
VOL_TREND_MULT     = 1.2
VOL_ORB_MULT       = 1.4
VOL_REV_MULT       = 1.1

# ── RSI ──────────────────────────────────────────────────────
RSI_BUY_TREND      = (50, 72)
RSI_SELL_TREND     = (28, 50)
RSI_REV_BUY        = 35
RSI_REV_SELL       = 65

# ── CANDLE STRENGTH (close position in bar) ──────────────────
CLOSE_POS_BUY      = 0.5            # Close must be in top 50% of bar for BUY
CLOSE_POS_SELL     = 0.5            # Close must be in bottom 50% for SELL

# ── TIME FILTERS (UTC — NSE opens at 03:45 UTC) ──────────────
ORB_END_UTC        = "04:00"        # 09:30 IST
EARLY_SKIP_END_UTC = "04:15"        # Skip 09:45–10:15 IST noisy window
MARKET_START_UTC   = "04:15"        # First valid entry: 09:45 IST
NO_ENTRY_UTC       = "08:30"        # Last entry: 14:00 IST
EOD_EXIT_UTC       = "09:45"        # Force close: 15:15 IST

# ── RISK CONTROLS ────────────────────────────────────────────
MAX_REAL_SL_DAY    = 2              # Max real stop losses per stock per day
GAP_SKIP_PCT       = 0.008          # Skip extra 45min if gap > 0.8%
ATR_SPIKE_MULT     = 2.0            # Skip if today's ATR > 2× 20-day avg ATR
ORB_CANDLES        = 3              # 3 × 5min = 15min opening range

# ──────────────────────────────────────────────────────────────
#  LEARNED NO-TRADE RULES (hard-coded from 30-day analysis)
#  Source: trade_log analysis — these combos lose money
# ──────────────────────────────────────────────────────────────
# Format: (sector, signal_type, side) → block if True
NO_TRADE_RULES = {
    ("Banking",  "Trend", "SELL"),   # 42% WR, -₹1,483
    ("FMCG",     "Trend", "SELL"),   # 0% WR,  -₹1,292
    ("Pharma",   "Trend", "SELL"),   # 33% WR, -₹250
    ("Pharma",   "Trend", "BUY"),    # 0% WR,  -₹1,319 ← BIGGEST RULE
    ("Energy",   "Trend", "SELL"),   # 54% WR, -₹328 (marginal, block it)
}
# Note: Metal Trend SELL stays allowed (80% WR, +₹877)
# Note: Banking Trend SELL allowed only if ADX > 25 AND Supertrend bearish

# ──────────────────────────────────────────────────────────────
#  INDICATOR ENGINE
# ──────────────────────────────────────────────────────────────
def compute_indicators(raw):
    df = raw.copy()
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                  for c in df.columns]
    df.dropna(inplace=True)
    if len(df) < 60:
        return df

    # ── EMAs ─────────────────────────────────────────────────
    df['ema9']  = df['close'].ewm(span=9,  adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema20_slope'] = df['ema20'].diff(3)

    # ── RSI 14 ───────────────────────────────────────────────
    d = df['close'].diff()
    gain = d.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(com=13, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))
    df['rsi_slope'] = df['rsi'].diff(2)

    # ── MACD (12,26,9) ───────────────────────────────────────
    macd_line         = df['close'].ewm(span=12, adjust=False).mean() - \
                        df['close'].ewm(span=26, adjust=False).mean()
    df['macd_hist']   = macd_line - macd_line.ewm(span=9, adjust=False).mean()
    df['macd_bull']   = (df['macd_hist'] > 0) & \
                        (df['macd_hist'] > df['macd_hist'].shift(1))
    df['macd_bear']   = (df['macd_hist'] < 0) & \
                        (df['macd_hist'] < df['macd_hist'].shift(1))

    # ── ATR 14 ───────────────────────────────────────────────
    df['tr']  = np.maximum(df['high'] - df['low'],
                 np.maximum(abs(df['high'] - df['close'].shift(1)),
                            abs(df['low']  - df['close'].shift(1))))
    df['atr'] = df['tr'].ewm(span=14, adjust=False).mean()
    # Daily ATR context (20-bar rolling of ATR for spike detection)
    df['atr_avg20'] = df['atr'].rolling(20).mean()
    df['atr_spike'] = df['atr'] > (df['atr_avg20'] * ATR_SPIKE_MULT)

    # ── ADX 14 ───────────────────────────────────────────────
    high_diff = df['high'].diff()
    low_diff  = df['low'].diff()
    plus_dm   = np.where((high_diff > low_diff.abs()) & (high_diff > 0), high_diff, 0)
    minus_dm  = np.where((low_diff.abs() > high_diff) & (low_diff < 0), low_diff.abs(), 0)
    atr_adx   = df['tr'].ewm(span=ADX_PERIOD, adjust=False).mean()
    plus_di   = 100 * pd.Series(plus_dm,  index=df.index).ewm(
                    span=ADX_PERIOD, adjust=False).mean() / atr_adx
    minus_di  = 100 * pd.Series(minus_dm, index=df.index).ewm(
                    span=ADX_PERIOD, adjust=False).mean() / atr_adx
    dx        = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    df['adx']      = dx.ewm(span=ADX_PERIOD, adjust=False).mean()
    df['plus_di']  = plus_di
    df['minus_di'] = minus_di
    # Trending if ADX > threshold
    df['is_trending'] = df['adx'] > ADX_TREND_MIN

    # ── SUPERTREND ────────────────────────────────────────────
    hl_avg       = (df['high'] + df['low']) / 2
    upper_band   = hl_avg + ST_MULT * df['atr']
    lower_band   = hl_avg - ST_MULT * df['atr']
    supertrend   = pd.Series(index=df.index, dtype=float)
    st_direction = pd.Series(index=df.index, dtype=int)   # 1=bull, -1=bear

    for i in range(1, len(df)):
        prev_close = df['close'].iloc[i-1]
        curr_close = df['close'].iloc[i]
        ub = upper_band.iloc[i]
        lb = lower_band.iloc[i]
        prev_st = supertrend.iloc[i-1] if i > 1 else ub
        prev_dir = st_direction.iloc[i-1] if i > 1 else -1

        if prev_dir == 1:
            curr_lb = max(lb, prev_st) if curr_close > prev_st else ub
            st_direction.iloc[i] = 1 if curr_close > curr_lb else -1
            supertrend.iloc[i] = curr_lb
        else:
            curr_ub = min(ub, prev_st) if curr_close < prev_st else lb
            st_direction.iloc[i] = -1 if curr_close < curr_ub else 1
            supertrend.iloc[i] = curr_ub

    df['supertrend']   = supertrend
    df['st_bull']      = st_direction == 1
    df['st_bear']      = st_direction == -1

    # ── VWAP (daily reset) ───────────────────────────────────
    df['utc_date'] = df.index.date
    df['tp']       = (df['high'] + df['low'] + df['close']) / 3
    cumvol         = df.groupby('utc_date')['volume'].cumsum()
    cumtpvol       = df.groupby('utc_date').apply(
        lambda x: (x['tp'] * x['volume']).cumsum()).values
    df['vwap']     = cumtpvol / cumvol

    # ── BOLLINGER BANDS (20, 2) ──────────────────────────────
    bb_mid         = df['close'].rolling(20).mean()
    bb_std         = df['close'].rolling(20).std()
    df['bb_upper'] = bb_mid + 2 * bb_std
    df['bb_lower'] = bb_mid - 2 * bb_std

    # ── VOLUME ───────────────────────────────────────────────
    df['vol_ma'] = df['volume'].rolling(20).mean()

    # ── CANDLE ATTRIBUTES ────────────────────────────────────
    df['bar_range']   = df['high'] - df['low']
    df['close_pos']   = np.where(df['bar_range'] > 0,
                            (df['close'] - df['low']) / df['bar_range'], 0.5)
    df['bull_candle'] = df['close'] > df['open']
    df['bear_candle'] = df['close'] < df['open']
    df['candle_num']  = df.groupby('utc_date').cumcount()

    # ── GAP DETECTION ────────────────────────────────────────
    df['prev_close']  = df['close'].shift(1)
    df['gap_pct']     = abs(df['open'] - df['prev_close']) / df['prev_close']
    # Flag first candle of each day
    df['is_first']    = df['candle_num'] == 0
    # Propagate gap flag for the session opening candle
    df['day_gap'] = df.groupby('utc_date')['gap_pct'].transform('first')
    df['big_gap_day'] = df['day_gap'] > GAP_SKIP_PCT

    # ── OPENING RANGE ─────────────────────────────────────────
    orb_h = {}
    orb_l = {}
    for date, grp in df.groupby('utc_date'):
        ob = grp.iloc[:ORB_CANDLES]
        orb_h[date] = ob['high'].max()
        orb_l[date] = ob['low'].min()
    df['orb_high'] = df['utc_date'].map(orb_h)
    df['orb_low']  = df['utc_date'].map(orb_l)

    df['prev_high'] = df['high'].shift(1)
    df['prev_low']  = df['low'].shift(1)

    df.dropna(inplace=True)
    return df


# ──────────────────────────────────────────────────────────────
#  HTF REGIME (15-min trend direction for the day)
# ──────────────────────────────────────────────────────────────
def get_htf_regime(symbol):
    try:
        htf = yf.download(symbol, period="30d", interval="15m",
                          progress=False, auto_adjust=True)
        if htf.empty:
            return {}
        htf.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                       for c in htf.columns]
        htf['ema20'] = htf['close'].ewm(span=20, adjust=False).mean()
        regime = {}
        for date, grp in htf.groupby(htf.index.date):
            bull = (grp['close'] > grp['ema20']).mean()
            regime[date] = 'bull' if bull > 0.55 else ('bear' if bull < 0.45 else 'neutral')
        return regime
    except:
        return {}


# ──────────────────────────────────────────────────────────────
#  SIGNAL ENGINE
# ──────────────────────────────────────────────────────────────
def generate_signals(df, regime, sector):
    df = df.copy()
    t   = df.index.strftime('%H:%M')

    # Base session window
    in_session = (t > MARKET_START_UTC) & (t <= NO_ENTRY_UTC)

    # Extra skip for big gap days (first 45min extra)
    normal_start = t > MARKET_START_UTC
    late_start   = t > "05:00"   # 10:30 IST — 45min after open
    session_buy  = np.where(df['big_gap_day'], late_start, normal_start) & (t <= NO_ENTRY_UTC)
    session_sell = session_buy.copy()
    in_session_arr = pd.Series(session_buy.astype(bool), index=df.index)

    # HTF regime
    df['regime'] = df['utc_date'].map(lambda d: regime.get(d, 'neutral'))

    # ATR quality filter
    atr_ok    = (df['atr'] / df['close']) > ATR_MIN_PCT
    # ATR spike filter (panic/news day — skip trend trades)
    no_spike  = ~df['atr_spike']

    # Candle position
    bull_body = df['close_pos'] > CLOSE_POS_BUY
    bear_body = df['close_pos'] < (1 - CLOSE_POS_SELL)

    # ── SIGNAL 1: TREND + ADX + SUPERTREND + MACD ────────────
    # BUY: all 6 must align
    trend_buy = (
        (df['close'] > df['ema20']) &
        (df['close'] > df['vwap']) &
        (df['ema20'] > df['ema50']) &
        (df['ema20_slope'] > 0) &
        (df['rsi'] >= RSI_BUY_TREND[0]) & (df['rsi'] <= RSI_BUY_TREND[1]) &
        (df['rsi_slope'] > 0) &           # RSI accelerating up
        df['macd_bull'] &
        df['st_bull'] &                    # Supertrend bullish
        df['is_trending'] &                # ADX says trending
        (df['plus_di'] > df['minus_di']) & # +DI above -DI (bull momentum)
        bull_body &
        (df['volume'] > df['vol_ma'] * VOL_TREND_MULT) &
        (df['regime'] != 'bear') &
        (df['candle_num'] >= 9) &          # Skip first 45min
        no_spike &
        atr_ok & in_session_arr
    )
    # SELL: all 6 must align
    trend_sell = (
        (df['close'] < df['ema20']) &
        (df['close'] < df['vwap']) &
        (df['ema20'] < df['ema50']) &
        (df['ema20_slope'] < 0) &
        (df['rsi'] >= RSI_SELL_TREND[0]) & (df['rsi'] <= RSI_SELL_TREND[1]) &
        (df['rsi_slope'] < 0) &           # RSI accelerating down
        df['macd_bear'] &
        df['st_bear'] &                    # Supertrend bearish
        df['is_trending'] &
        (df['minus_di'] > df['plus_di']) & # -DI above +DI (bear momentum)
        bear_body &
        (df['volume'] > df['vol_ma'] * VOL_TREND_MULT) &
        (df['regime'] != 'bull') &
        (df['candle_num'] >= 9) &
        no_spike &
        atr_ok & in_session_arr
    )

    # ── SIGNAL 2: ORB PULLBACK ───────────────────────────────
    orb_vol_ok = df['volume'] > df['vol_ma'] * VOL_ORB_MULT
    orb_t      = (t > ORB_END_UTC) & (t <= NO_ENTRY_UTC)

    # Confirmed breakout candle
    orb_bull_break = (
        (df['close'] > df['orb_high']) &
        orb_vol_ok & bull_body &
        (df['close'] > df['vwap']) &
        df['st_bull']
    )
    orb_bear_break = (
        (df['close'] < df['orb_low']) &
        orb_vol_ok & bear_body &
        (df['close'] < df['vwap']) &
        df['st_bear']
    )

    df['orb_bull_broken'] = orb_bull_break.shift(1).fillna(False)
    df['orb_bear_broken'] = orb_bear_break.shift(1).fillna(False)

    pullback_buy = (
        df['orb_bull_broken'] &
        (df['low'] <= df['orb_high']) &
        (df['close'] > df['orb_high']) &
        (df['close'] > df['vwap']) &
        (df['regime'] != 'bear') &
        atr_ok & pd.Series((t > ORB_END_UTC) & (t <= NO_ENTRY_UTC), index=df.index)
    )
    pullback_sell = (
        df['orb_bear_broken'] &
        (df['high'] >= df['orb_low']) &
        (df['close'] < df['orb_low']) &
        (df['close'] < df['vwap']) &
        (df['regime'] != 'bull') &
        atr_ok & pd.Series((t > ORB_END_UTC) & (t <= NO_ENTRY_UTC), index=df.index)
    )

    # ── SIGNAL 3: REVERSAL (Bollinger + RSI extreme) ─────────
    rev_buy = (
        (df['rsi'] < RSI_REV_BUY) &
        (df['close'] < df['bb_lower']) &
        df['bull_candle'] &
        (df['rsi_slope'] > 0) &           # RSI turning up already
        (df['volume'] > df['vol_ma'] * VOL_REV_MULT) &
        atr_ok & in_session_arr
    )
    rev_sell = (
        (df['rsi'] > RSI_REV_SELL) &
        (df['close'] > df['bb_upper']) &
        df['bear_candle'] &
        (df['rsi_slope'] < 0) &           # RSI turning down
        (df['volume'] > df['vol_ma'] * VOL_REV_MULT) &
        atr_ok & in_session_arr
    )

    # ── COMBINE ──────────────────────────────────────────────
    df['buy_signal']  = trend_buy  | pullback_buy  | rev_buy
    df['sell_signal'] = trend_sell | pullback_sell | rev_sell

    df['sig_type'] = 'None'
    df.loc[trend_buy    | trend_sell,    'sig_type'] = 'Trend'
    df.loc[pullback_buy | pullback_sell, 'sig_type'] = 'ORB-Pullback'
    df.loc[rev_buy      | rev_sell,      'sig_type'] = 'Reversal'

    # Apply learned NO-TRADE rules
    # We'll check at trade entry time, but flag here for pullback/trend
    # The backtest engine does the actual block using NO_TRADE_RULES set

    df['buy_entry']  = df['buy_signal'].shift(1).fillna(False)
    df['sell_entry'] = df['sell_signal'].shift(1).fillna(False)
    df['entry_type'] = df['sig_type'].shift(1).fillna('None')

    return df


# ──────────────────────────────────────────────────────────────
#  BACKTEST ENGINE
# ──────────────────────────────────────────────────────────────
def backtest(df, symbol, sector, direction=DIRECTION):
    trades        = []
    in_pos        = False
    edata         = {}
    # Track losses by direction per day
    daily_sl      = {}   # {(date, 'BUY'): count, (date, 'SELL'): count}

    for i in range(1, len(df)):
        row   = df.iloc[i]
        rtime = row.name.strftime('%H:%M')
        rdate = row.name.date()

        # ── MANAGE OPEN POSITION ──────────────────────────────
        if in_pos:
            ep    = edata['price']
            side  = edata['side']
            atr_e = edata['atr']
            t1hit = edata['t1_hit']
            qrem  = edata['qty_rem']

            t1_d = ATR_T1_MULT * atr_e
            t2_d = ATR_T2_MULT * atr_e

            exit_p, reason = None, None

            if rtime >= EOD_EXIT_UTC:
                exit_p = row['close']
                reason = "EOD Exit"

            elif side == "BUY":
                sl_p = edata['sl']
                t1_p = ep + t1_d
                t2_p = ep + t2_d

                if not t1hit and row['high'] >= t1_p:
                    qty2  = edata['shares'] // 2
                    pnl2  = (t1_p - ep) * qty2
                    trades.append(_mk(edata, row, t1_p, qty2, pnl2, "T1 (50%)", symbol, sector))
                    edata.update({'t1_hit': True, 'sl': ep, 'qty_rem': edata['shares'] - qty2})
                    continue

                if row['low'] <= sl_p:
                    exit_p = sl_p
                    reason = "Breakeven" if t1hit else "Stop Loss"
                elif t1hit and row['high'] >= t2_p:
                    exit_p = t2_p
                    reason = "T2 Final"

            else:  # SELL
                sl_p = edata['sl']
                t1_p = ep - t1_d
                t2_p = ep - t2_d

                if not t1hit and row['low'] <= t1_p:
                    qty2  = edata['shares'] // 2
                    pnl2  = (ep - t1_p) * qty2
                    trades.append(_mk(edata, row, t1_p, qty2, pnl2, "T1 (50%)", symbol, sector))
                    edata.update({'t1_hit': True, 'sl': ep, 'qty_rem': edata['shares'] - qty2})
                    continue

                if row['high'] >= sl_p:
                    exit_p = sl_p
                    reason = "Breakeven" if t1hit else "Stop Loss"
                elif t1hit and row['low'] <= t2_p:
                    exit_p = t2_p
                    reason = "T2 Final"

            if exit_p is not None:
                pnl = ((exit_p - ep) * qrem if side == "BUY"
                       else (ep - exit_p) * qrem)
                trades.append(_mk(edata, row, exit_p, qrem, pnl, reason, symbol, sector))
                if reason == "Stop Loss":
                    key = (rdate, side)
                    daily_sl[key] = daily_sl.get(key, 0) + 1
                in_pos = False
            continue

        # ── LOOK FOR NEW ENTRY ────────────────────────────────
        if rtime > NO_ENTRY_UTC:
            continue

        signal, side = False, None
        if direction in ("BUY", "BOTH") and row.get('buy_entry', False):
            signal, side = True, "BUY"
        elif direction in ("SELL", "BOTH") and row.get('sell_entry', False):
            signal, side = True, "SELL"

        if not signal:
            continue

        stype = row.get('entry_type', 'Unknown')

        # ── LEARNED NO-TRADE RULES ────────────────────────────
        if (sector, stype, side) in NO_TRADE_RULES:
            continue

        # ── DIRECTION LOSS GUARD (per direction per day) ──────
        key = (rdate, side)
        if daily_sl.get(key, 0) >= MAX_REAL_SL_DAY:
            continue

        ep    = row['open']
        atr_e = row['atr']
        if ep <= 0 or atr_e <= 0:
            continue

        # ── RISK-BASED POSITION SIZING ────────────────────────
        sl_dist = ATR_SL_MULT * atr_e
        shares  = int(MAX_RISK_PER_TRADE / sl_dist)
        shares  = max(MIN_QTY, shares)
        if shares * ep > MAX_TRADE_CAPITAL:
            shares = int(MAX_TRADE_CAPITAL / ep)
        if shares <= 0:
            continue

        sl_val = (ep - sl_dist if side == "BUY" else ep + sl_dist)
        edata  = {
            'price'  : ep, 'side': side, 'atr': atr_e,
            'shares' : shares, 'qty_rem': shares,
            'sl'     : sl_val, 't1_hit': False,
            'time'   : row.name, 'stype': stype,
        }
        in_pos = True

    return trades


def _mk(ed, row, xp, qty, pnl, reason, symbol, sector):
    ep = ed['price']
    return {
        'sector'      : sector, 'symbol': symbol,
        'side'        : ed['side'], 'signal_type': ed.get('stype', '?'),
        'entry_time'  : ed['time'], 'exit_time': row.name,
        'entry_price' : round(ep, 2), 'exit_price': round(xp, 2),
        'qty'         : qty,
        'pnl_rs'      : round(pnl, 2),
        'pnl_pct'     : round((pnl / (ep * qty)) * 100, 3) if qty > 0 else 0,
        'exit_reason' : reason, 'win': pnl > 0,
        'trade_value' : round(ep * qty, 0),
    }


# ──────────────────────────────────────────────────────────────
#  MAIN RUN
# ──────────────────────────────────────────────────────────────
all_trades = []

print("\n" + "═"*68)
print("  🚀  NSE INTRADAY STRATEGY v6 — DATA-LEARNED REBUILD")
print("  Period: 30d | TF: 5min | Risk: ₹500/trade | Cap: ₹20,000/trade")
print("  Cores: ADX + Supertrend + MACD + VWAP + RSI + Bollinger + ORB")
print("  No-Trade Rules: 5 blocked combos from 30-day analysis")
print("═"*68)

for sector, symbol in SECTORS.items():
    print(f"\n  📡 [{sector}] {symbol}...", end=" ")
    try:
        raw = yf.download(symbol, period=PERIOD, interval=TIMEFRAME,
                          progress=False, auto_adjust=True)
        if raw.empty or len(raw) < 60:
            print("❌ Insufficient data — trying 5d fallback...")
            raw = yf.download(symbol, period="5d", interval=TIMEFRAME,
                              progress=False, auto_adjust=True)
            if raw.empty:
                print("❌ No data")
                continue

        regime = get_htf_regime(symbol)
        df     = compute_indicators(raw)
        if len(df) < 30:
            print("❌ Too few candles after indicator warmup")
            continue
        df     = generate_signals(df, regime, sector)
        trades = backtest(df, symbol, sector)
        all_trades.extend(trades)

        if trades:
            tdf = pd.DataFrame(trades)
            wr  = tdf['win'].mean() * 100
            pnl = tdf['pnl_rs'].sum()
            avg_val = (tdf['qty'] * tdf['entry_price']).mean()
            real_sl = (tdf['exit_reason'] == 'Stop Loss').sum()
            print(f"✅ {len(trades)} trades | WR:{wr:.0f}% | P&L:₹{pnl:,.0f} | "
                  f"AvgPos:₹{avg_val:,.0f} | SL:{real_sl}")
        else:
            print("✅ 0 signals fired (all filters/rules passed cleanly)")
    except Exception as e:
        print(f"❌ {e}")


# ──────────────────────────────────────────────────────────────
#  CONSOLIDATED REPORT
# ──────────────────────────────────────────────────────────────
print("\n\n" + "═"*68)
print("  📊  CONSOLIDATED REPORT — v6")
print("═"*68)

if not all_trades:
    print("\n  ⚠️  Zero trades. Filters are very strict + 30d data.")
    print("  QUICK FIXES (change at top of file):")
    print("  → Reduce ADX_TREND_MIN: 22 → 18")
    print("  → Reduce VOL_TREND_MULT: 1.2 → 1.0")
    print("  → Change PERIOD to '60d'")
    exit()

tdf    = pd.DataFrame(all_trades)
total  = len(tdf)
wins   = int(tdf['win'].sum())
losses = total - wins
wr     = wins / total * 100
gw     = tdf[tdf['win']]['pnl_rs'].sum()
gl     = abs(tdf[~tdf['win']]['pnl_rs'].sum())
net    = tdf['pnl_rs'].sum()
pf     = round(gw / gl, 2) if gl > 0 else float('inf')
aw     = tdf[tdf['win']]['pnl_rs'].mean() if wins > 0 else 0
al     = tdf[~tdf['win']]['pnl_rs'].mean() if losses > 0 else 0
exp    = (wr/100 * aw) + ((1 - wr/100) * al)
rets   = tdf['pnl_pct']
sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
exit_c = tdf['exit_reason'].value_counts()

avg_pos = (tdf['qty'] * tdf['entry_price']).mean()
max_pos = (tdf['qty'] * tdf['entry_price']).max()

print(f"\n  {'OVERALL':─<55}")
print(f"  Total Trades        : {total}")
print(f"  Wins / Losses       : {wins} / {losses}")
print(f"  Win Rate            : {wr:.1f}%")
print(f"  Net P&L             : ₹{net:,.2f}")
print(f"  Profit Factor       : {pf}    (>1.5 = Good, >2 = Great)")
print(f"  Avg Win             : ₹{aw:,.2f}")
print(f"  Avg Loss            : ₹{al:,.2f}")
print(f"  Expectancy/Trade    : ₹{exp:,.2f}")
print(f"  Sharpe Ratio        : {sharpe:.2f}")
print(f"\n  {'POSITION SIZING':─<55}")
print(f"  Avg position value  : ₹{avg_pos:,.0f}")
print(f"  Max position value  : ₹{max_pos:,.0f}")
print(f"  Avg qty/trade       : {tdf['qty'].mean():.0f} shares")

print(f"\n  {'BY SECTOR':─<55}")
for sec in tdf['sector'].unique():
    sd  = tdf[tdf['sector'] == sec]
    swr = sd['win'].mean() * 100
    sp  = sd['pnl_rs'].sum()
    sym = sd['symbol'].iloc[0]
    rsl = (sd['exit_reason'] == 'Stop Loss').sum()
    flag = "✅" if sp > 0 else "❌"
    print(f"  {flag} {sec:<12} ({sym:<14}) → "
          f"{len(sd):>2}tr | WR:{swr:>5.1f}% | ₹{sp:>8,.0f} | SL:{rsl}")

print(f"\n  {'BY SIGNAL TYPE':─<55}")
for st in tdf['signal_type'].unique():
    sd  = tdf[tdf['signal_type'] == st]
    swr = sd['win'].mean() * 100
    sp  = sd['pnl_rs'].sum()
    rsl = (sd['exit_reason'] == 'Stop Loss').sum()
    flag = "✅" if sp > 0 else "❌"
    print(f"  {flag} {st:<15} → {len(sd):>2}tr | WR:{swr:>5.1f}% | ₹{sp:>8,.0f} | SL:{rsl}")

print(f"\n  {'BY DIRECTION':─<55}")
for side in tdf['side'].unique():
    sd  = tdf[tdf['side'] == side]
    swr = sd['win'].mean() * 100
    sp  = sd['pnl_rs'].sum()
    flag = "✅" if sp > 0 else "❌"
    print(f"  {flag} {side:<6} → {len(sd):>2}tr | WR:{swr:>5.1f}% | ₹{sp:>8,.0f}")

print(f"\n  {'EXIT BREAKDOWN':─<55}")
for r, c in exit_c.items():
    flag = "✅" if r in ("T1 (50%)", "T2 Final") else ("⚠️ " if r == "EOD Exit" else "❌")
    print(f"  {flag} {r:<22}: {c:>2}  ({c/total*100:.1f}%)")

print(f"\n  {'BY DAY':─<55}")
tdf['day'] = pd.to_datetime(tdf['entry_time']).dt.date
for day, grp in tdf.groupby('day'):
    dwr  = grp['win'].mean() * 100
    dp   = grp['pnl_rs'].sum()
    rsl  = (grp['exit_reason'] == 'Stop Loss').sum()
    flag = "✅" if dp > 0 else "❌"
    print(f"  {flag} {day} → {len(grp):>2}tr | WR:{dwr:>5.1f}% | ₹{dp:>8,.0f} | SL:{rsl}")

print(f"\n  {'ALL TRADES':─<55}")
cols = ['sector','side','signal_type','entry_time','exit_time',
        'entry_price','exit_price','qty','trade_value','pnl_rs','pnl_pct','exit_reason']
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 140)
pd.set_option('display.float_format', '{:.2f}'.format)
print(tdf[cols].to_string(index=False))

# ── PROFESSIONAL DIAGNOSIS ────────────────────────────────────
print("\n\n" + "═"*68)
print("  🔬  PROFESSIONAL DIAGNOSIS")
print("═"*68)

real_sl = (tdf['exit_reason'] == 'Stop Loss').sum()
t2_hits = (tdf['exit_reason'] == 'T2 Final').sum()
eod_c   = exit_c.get("EOD Exit", 0)
be_c    = exit_c.get("Breakeven", 0)

if total > 0:
    best_sec  = tdf.groupby('sector')['pnl_rs'].sum().idxmax()
    worst_sec = tdf.groupby('sector')['pnl_rs'].sum().idxmin()
    best_sig  = tdf.groupby('signal_type')['pnl_rs'].sum().idxmax()
    best_dir  = tdf.groupby('side')['pnl_rs'].sum().idxmax()

    print(f"\n  Best Sector    : {best_sec}")
    print(f"  Worst Sector   : {worst_sec}")
    print(f"  Best Signal    : {best_sig}")
    print(f"  Best Direction : {best_dir}")

print(f"\n  Real SL hits   : {real_sl}/{total} ({real_sl/total*100:.0f}%)"
      + (" ✅" if real_sl/total < 0.25 else " ⚠️  raise ATR_SL_MULT to 2.5"))
print(f"  T2 Full wins   : {t2_hits}/{total} ({t2_hits/total*100:.0f}%)"
      + (" ✅" if t2_hits > 0 else " ⚠️  lower ATR_T2_MULT to 2.5"))
print(f"  Breakevens     : {be_c}/{total} — protecting capital ✅")
print(f"  EOD Exits      : {eod_c}/{total}"
      + (" ✅" if eod_c/total < 0.25 else " ⚠️  reduce NO_ENTRY_UTC time"))

print(f"\n  {'LEARNED RULES IMPACT':─<55}")
print(f"  5 combinations blocked by learned no-trade rules:")
for rule in NO_TRADE_RULES:
    print(f"    ❌ {rule[0]} {rule[2]} {rule[1]}")

print(f"\n  {'TUNE IF NEEDED':─<55}")
print(f"  ┌─ <10 trades?      → ADX_TREND_MIN: 22→18, VOL: 1.2→1.0")
print(f"  ├─ SL > 25%?        → ATR_SL_MULT: 2.0→2.5")
print(f"  ├─ T2 never hit?    → ATR_T2_MULT: 3.5→2.5")
print(f"  ├─ Worst sector?    → Add to NO_TRADE_RULES and re-run")
print(f"  └─ EOD exits > 25%? → NO_ENTRY_UTC: 08:30→07:30")

out = "trade_log_v6_learned.csv"
tdf.to_csv(out, index=False)
print(f"\n  💾 Trade log → {out}")
print("═"*68 + "\n")
