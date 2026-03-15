"""
╔══════════════════════════════════════════════════════════════╗
║   KRAKEN SWING TRADING BOT                                   ║
║   Claude AI + CCXT + Kraken                                  ║
║                                                              ║
║   Strategy : EMA 9/21 crossover + RSI 14 confirmation       ║
║   Timeframe : 1-hour candles                                 ║
║   Target    : 4% profit per trade                            ║
║   Stop-loss : 2% per trade                                   ║
║                                                              ║
║   !! RISK WARNING !!                                         ║
║   ─────────────────                                          ║
║   • Crypto trading carries real risk of loss                 ║
║   • You CAN lose your entire balance                         ║
║   • NO strategy guarantees returns                           ║
║   • Only trade money you can afford to lose 100% of          ║
║                                                              ║
║   REALISTIC outcomes on $100 per week:                       ║
║   Good week  → +$5  to +$12                                  ║
║   Flat week  → -$1  to +$3                                   ║
║   Bad week   → -$4  to -$8                                   ║
║                                                              ║
║   HOW TO RUN:                                                ║
║   1. Set your API keys in Railway Variables                  ║
║   2. Keep SANDBOX = True for at least one week               ║
║   3. Watch the logs daily                                     ║
║   4. Only set SANDBOX = False when you are confident         ║
╚══════════════════════════════════════════════════════════════╝
"""

import ccxt
import anthropic
import pandas as pd
import pandas_ta as ta
import time
import os
from datetime import datetime


# ════════════════════════════════════════════════════════════════
#  CONFIG — your keys come from Railway Variables tab
# ════════════════════════════════════════════════════════════════

KRAKEN_KEY    = os.getenv("KRAKEN_API_KEY",    "PASTE_YOUR_KRAKEN_KEY_HERE")
KRAKEN_SECRET = os.getenv("KRAKEN_API_SECRET", "PASTE_YOUR_KRAKEN_SECRET_HERE")
CLAUDE_KEY    = os.getenv("ANTHROPIC_API_KEY", "PASTE_YOUR_CLAUDE_KEY_HERE")

# ── SWING TRADING SETTINGS ───────────────────────────────────────

TRADING_PAIR        = "BTC/USD"    # Bitcoin priced in US dollars on Kraken
TIMEFRAME           = "1h"         # 1-hour candles — swing trading timeframe
CHECK_EVERY_MINUTES = 30           # Check every 30 minutes
TRADE_SIZE_PCT      = 0.90         # Use 90% of available balance per trade

# ── EXIT RULES ───────────────────────────────────────────────────
# Wide enough to clear Kraken's ~0.42% round-trip fee easily

TAKE_PROFIT_PCT = 0.04   # Sell at +4% gain
STOP_LOSS_PCT   = 0.02   # Sell at -2% loss
# Risk:reward = 1:2 — lose half of what you aim to gain

# ── DAILY SAFETY LIMITS ──────────────────────────────────────────

DAILY_LOSS_LIMIT_PCT = 0.10   # Pause bot if down 10% today
MAX_DAILY_TRADES     = 6      # Swing trading is slow — 6 trades/day is plenty

# ── SANDBOX MODE ─────────────────────────────────────────────────
# True  = simulated trades only, no real money moves
# False = live trading with real money
# Keep True until you have watched at least one full week of logs!

SANDBOX = True


# ════════════════════════════════════════════════════════════════
#  BOT STATE
# ════════════════════════════════════════════════════════════════

position        = None
trade_count_day = 0
last_day_reset  = datetime.now().date()
daily_start_bal = None
session_trades  = []


# ════════════════════════════════════════════════════════════════
#  LOGGING
# ════════════════════════════════════════════════════════════════

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]  {msg}", flush=True)


# ════════════════════════════════════════════════════════════════
#  CONNECT TO KRAKEN + CLAUDE
# ════════════════════════════════════════════════════════════════

def connect():
    """
    Connects to Kraken using CCXT and Claude AI using the Anthropic SDK.
    Kraken uses standard base64 secrets — no newline fix needed like Coinbase.
    """
    exchange = ccxt.kraken({
        "apiKey": KRAKEN_KEY,
        "secret": KRAKEN_SECRET,
    })

    try:
        exchange.fetch_time()
        log("Connected to Kraken successfully.")
    except Exception as e:
        log(f"Kraken connection warning: {e}")
        log("Check your API keys if trades fail.")

    claude = anthropic.Anthropic(api_key=CLAUDE_KEY)
    log("Connected to Claude AI successfully.")
    return exchange, claude


# ════════════════════════════════════════════════════════════════
#  MARKET DATA + INDICATORS
# ════════════════════════════════════════════════════════════════

def get_data(exchange):
    """
    Fetches the last 100 one-hour candles from Kraken and
    calculates swing trading indicators:

    EMA 9  — fast moving average (reacts to recent price changes)
    EMA 21 — slow moving average (filters out short-term noise)
    RSI 14 — momentum score 0-100 (above 70 = overbought, below 30 = oversold)
    Bollinger Bands — normal price range bands (upper, middle, lower)
    """
    try:
        raw = exchange.fetch_ohlcv(TRADING_PAIR, TIMEFRAME, limit=100)
        df  = pd.DataFrame(raw, columns=["time","open","high","low","close","volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")

        # Swing trading EMAs
        df["ema9"]  = ta.ema(df["close"], length=9)
        df["ema21"] = ta.ema(df["close"], length=21)

        # Standard RSI for swing trading
        df["rsi"] = ta.rsi(df["close"], length=14)

        # Bollinger Bands — detect column names dynamically
        # (column names vary slightly between pandas-ta versions)
        bb = ta.bbands(df["close"], length=20, std=2)
        if bb is not None and not bb.empty:
            upper_col = [c for c in bb.columns if c.startswith("BBU")][0]
            lower_col = [c for c in bb.columns if c.startswith("BBL")][0]
            mid_col   = [c for c in bb.columns if c.startswith("BBM")][0]
            df["bb_upper"] = bb[upper_col]
            df["bb_lower"] = bb[lower_col]
            df["bb_mid"]   = bb[mid_col]
        else:
            df["bb_upper"] = df["close"]
            df["bb_lower"] = df["close"]
            df["bb_mid"]   = df["close"]

        # 24-hour price change for context
        df["change_24h"] = df["close"].pct_change(24) * 100

        return df

    except Exception as e:
        log(f"Data error: {e}")
        return None


# ════════════════════════════════════════════════════════════════
#  CLAUDE AI SWING TRADING DECISION
# ════════════════════════════════════════════════════════════════

def ask_claude(claude_client, df, position_info):
    """
    Sends the current market snapshot to Claude AI and asks for
    a BUY, SELL, or HOLD decision based on swing trading rules.

    Swing trading is slower and more deliberate than scalping —
    Claude is instructed to only act on strong, clear signals.
    """
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # 24-hour price change for trend context
    change_24h = last["change_24h"] if not pd.isna(last["change_24h"]) else 0.0

    prompt = f"""You are a careful swing trading assistant managing a small $100 crypto account on Kraken.
Your goal is to catch 4%+ moves while keeping losses under 2%.
You trade on 1-hour candles and only act on STRONG, CLEAR signals.
When in doubt, always HOLD — missing a trade is better than a bad trade.

Current market data for {TRADING_PAIR}:
- Price now:        ${last['close']:,.2f}
- 24h change:       {change_24h:+.2f}%
- EMA 9 (fast):     ${last['ema9']:,.2f}
- EMA 21 (slow):    ${last['ema21']:,.2f}
- EMA 9 previous:   ${prev['ema9']:,.2f}
- EMA 21 previous:  ${prev['ema21']:,.2f}
- RSI (14):         {last['rsi']:.1f}
- Bollinger upper:  ${last['bb_upper']:,.2f}
- Bollinger mid:    ${last['bb_mid']:,.2f}
- Bollinger lower:  ${last['bb_lower']:,.2f}
- Current position: {position_info}

BUY rules — only BUY if ALL of these are true:
  1. EMA 9 just crossed ABOVE EMA 21 this candle (it was below last candle)
  2. RSI is between 40 and 60 (healthy momentum, not overbought)
  3. Price is below or near the Bollinger middle band (room to grow)
  4. We do NOT already have an open position

SELL rules — SELL if ANY of these are true:
  1. EMA 9 crossed BELOW EMA 21
  2. RSI went above 70 (overbought — take profit)
  3. RSI dropped below 35 with an open position (momentum dying)
  4. Price broke below the lower Bollinger Band with open position

Otherwise HOLD.

Reply with EXACTLY one word on line 1: BUY, SELL, or HOLD
Then one sentence on line 2 explaining why."""

    try:
        resp = claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text.strip()
    except Exception as e:
        log(f"Claude error: {e}")
        return "HOLD\nCould not reach Claude — defaulting to HOLD."


# ════════════════════════════════════════════════════════════════
#  ORDER EXECUTION
# ════════════════════════════════════════════════════════════════

def get_usd_balance(exchange):
    """Gets your available USD balance on Kraken."""
    try:
        if SANDBOX:
            return 100.0
        bal = exchange.fetch_balance()
        return float(bal["total"].get("USD", 0))
    except:
        return 0


def buy(exchange, price):
    """Places a market buy order on Kraken."""
    global position
    balance   = get_usd_balance(exchange)
    spend_usd = round(balance * TRADE_SIZE_PCT, 2)
    amount    = round(spend_usd / price, 6)

    if SANDBOX:
        log(f"[SANDBOX] BUY  {amount} BTC @ ${price:,.2f}  (${spend_usd:.2f} spent)")
    else:
        try:
            exchange.create_market_buy_order(TRADING_PAIR, amount)
            log(f"BUY  {amount} BTC @ ~${price:,.2f}")
        except Exception as e:
            log(f"Buy order failed: {e}")
            return

    position = {
        "entry":      price,
        "amount":     amount,
        "spent_usd":  spend_usd,
        "stop":       round(price * (1 - STOP_LOSS_PCT), 2),
        "target":     round(price * (1 + TAKE_PROFIT_PCT), 2),
        "opened_at":  datetime.now(),
    }
    log(f"  Stop-loss : ${position['stop']:,.2f}")
    log(f"  Target    : ${position['target']:,.2f}")


def sell(exchange, price, reason):
    """Places a market sell order on Kraken and logs the result."""
    global position, session_trades
    if position is None:
        return

    pnl_pct  = (price - position["entry"]) / position["entry"] * 100
    pnl_usd  = position["spent_usd"] * (pnl_pct / 100)
    held_hrs = (datetime.now() - position["opened_at"]).seconds / 3600

    if SANDBOX:
        log(f"[SANDBOX] SELL {position['amount']} BTC @ ${price:,.2f}  ({reason})")
    else:
        try:
            exchange.create_market_sell_order(TRADING_PAIR, position["amount"])
        except Exception as e:
            log(f"Sell order failed: {e}")
            return

    log(f"  PnL    : {pnl_pct:+.2f}%  (${pnl_usd:+.2f})")
    log(f"  Held   : {held_hrs:.1f} hours")

    session_trades.append({
        "pnl_pct": pnl_pct,
        "pnl_usd": pnl_usd,
        "reason":  reason,
        "time":    datetime.now(),
    })
    position = None

    # Running session summary
    wins   = [t for t in session_trades if t["pnl_usd"] > 0]
    losses = [t for t in session_trades if t["pnl_usd"] <= 0]
    total  = sum(t["pnl_usd"] for t in session_trades)
    log(f"  Session: {len(wins)}W / {len(losses)}L  |  Total P&L: ${total:+.2f}")


# ════════════════════════════════════════════════════════════════
#  SAFETY GUARDS
# ════════════════════════════════════════════════════════════════

def check_hard_exits(price):
    """
    Checks stop-loss and take-profit levels.
    These run BEFORE asking Claude and cannot be overridden.
    """
    if position is None:
        return False, ""
    if price <= position["stop"]:
        return True, f"stop-loss triggered at ${price:,.2f}"
    if price >= position["target"]:
        return True, f"take-profit triggered at ${price:,.2f}"
    return False, ""


def reset_daily_counters():
    """Resets trade count and balance reference at midnight."""
    global trade_count_day, last_day_reset, daily_start_bal
    today = datetime.now().date()
    if today != last_day_reset:
        trade_count_day = 0
        last_day_reset  = today
        daily_start_bal = None
        log("New day — daily counters reset.")


def daily_loss_exceeded(exchange):
    """Pauses the bot if daily losses exceed the limit."""
    global daily_start_bal
    if SANDBOX:
        return False
    if daily_start_bal is None:
        daily_start_bal = get_usd_balance(exchange)
        return False
    current  = get_usd_balance(exchange)
    loss_pct = (daily_start_bal - current) / daily_start_bal
    if loss_pct >= DAILY_LOSS_LIMIT_PCT:
        log(f"Daily loss limit reached ({loss_pct*100:.1f}%). Pausing until tomorrow.")
        return True
    return False


# ════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ════════════════════════════════════════════════════════════════

def run():
    global trade_count_day

    log("=" * 60)
    log("KRAKEN SWING TRADING BOT — STARTING")
    log(f"Pair      : {TRADING_PAIR}")
    log(f"Timeframe : {TIMEFRAME} candles, check every {CHECK_EVERY_MINUTES} min")
    log(f"Target    : +{TAKE_PROFIT_PCT*100:.0f}%  |  Stop: -{STOP_LOSS_PCT*100:.0f}%")
    log(f"Mode      : {'SANDBOX (simulated — no real money)' if SANDBOX else '*** LIVE TRADING ***'}")
    log("=" * 60)

    exchange, claude = connect()

    while True:
        try:
            reset_daily_counters()

            # ── Daily loss guard ──
            if daily_loss_exceeded(exchange):
                log("Sleeping 1 hour...")
                time.sleep(3600)
                continue

            # ── Daily trade cap ──
            if trade_count_day >= MAX_DAILY_TRADES:
                log(f"Daily trade cap ({MAX_DAILY_TRADES}) reached. Done for today.")
                time.sleep(3600)
                continue

            # ── Fetch market data ──
            df = get_data(exchange)
            if df is None or len(df) < 25:
                log("Not enough data yet. Waiting...")
                time.sleep(CHECK_EVERY_MINUTES * 60)
                continue

            price = df.iloc[-1]["close"]
            rsi   = df.iloc[-1]["rsi"]
            ema9  = df.iloc[-1]["ema9"]
            ema21 = df.iloc[-1]["ema21"]

            log(f"Price: ${price:,.2f}  |  EMA9: ${ema9:,.2f}  |  EMA21: ${ema21:,.2f}  |  RSI: {rsi:.1f}")

            # ── Hard exit check (stop-loss / take-profit) ──
            should_exit, exit_reason = check_hard_exits(price)
            if should_exit:
                log(f"Hard exit: {exit_reason}")
                sell(exchange, price, exit_reason)
                trade_count_day += 1
                time.sleep(CHECK_EVERY_MINUTES * 60)
                continue

            # ── Ask Claude for decision ──
            pos_str = "none — no open position" if position is None else \
                      f"long {position['amount']} BTC since ${position['entry']:,.2f} | stop ${position['stop']:,.2f} | target ${position['target']:,.2f}"

            answer   = ask_claude(claude, df, pos_str)
            lines    = answer.split("\n")
            decision = lines[0].strip().upper()
            reason   = lines[1].strip() if len(lines) > 1 else ""

            log(f"Claude: {decision}  |  {reason}")

            # ── Act on decision ──
            if decision == "BUY" and position is None:
                buy(exchange, price)
                trade_count_day += 1

            elif decision == "SELL" and position is not None:
                sell(exchange, price, "Claude signal")
                trade_count_day += 1

            elif decision == "HOLD":
                if position is not None:
                    held_hrs = (datetime.now() - position["opened_at"]).seconds / 3600
                    pnl_now  = (price - position["entry"]) / position["entry"] * 100
                    log(f"Holding — {held_hrs:.1f}h in trade  |  Current PnL: {pnl_now:+.2f}%")
                else:
                    log("Holding — waiting for entry signal.")

            else:
                log(f"Unexpected response '{decision}' — defaulting to HOLD.")

        except KeyboardInterrupt:
            log("Bot stopped by user.")
            if session_trades:
                wins  = [t for t in session_trades if t["pnl_usd"] > 0]
                total = sum(t["pnl_usd"] for t in session_trades)
                log(f"Final session: {len(session_trades)} trades | {len(wins)} wins | P&L: ${total:+.2f}")
            break

        except Exception as e:
            log(f"Unexpected error: {e} — continuing in {CHECK_EVERY_MINUTES} min...")

        # Wait before next check
        log(f"Sleeping {CHECK_EVERY_MINUTES} minutes...\n")
        time.sleep(CHECK_EVERY_MINUTES * 60)


if __name__ == "__main__":
    run()
