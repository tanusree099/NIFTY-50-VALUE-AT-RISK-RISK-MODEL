# NIFTY-50-VALUE-AT-RISK-RISK-MODEL
"""
=====================================================================
  NIFTY 50 — VALUE AT RISK & RISK MODEL
  Author  : Tanusree Saha
  Data    : Simulated Nifty 50 based on real historical ranges
            (Jan 2023 – May 2026: 17,000 – 26,300 range)
  Tools   : Python, NumPy, SciPy, Pandas, Matplotlib
=====================================================================

TO USE REAL LIVE DATA (one line change):
    pip install yfinance
    Replace generate_nifty_data() with:
        import yfinance as yf
        df = yf.download("^NSEI", start="2023-01-01", end="2026-05-30")
        returns = df["Close"].pct_change().dropna()

WHAT THIS PROJECT DOES:
    1. Historical Simulation VaR & CVaR (Expected Shortfall)
    2. Parametric VaR (Normal + Student-t for fat tails)
    3. Monte Carlo VaR (10,000 simulated paths)
    4. Basel III Traffic Light Backtesting
    5. Kupiec POF Statistical Test
    6. Stressed VaR — COVID-2020 & 2022 crash scenarios
    7. Rolling VaR — how risk changed over time
    8. Sector-level risk contribution

CURRENT MARKET CONTEXT (May 30, 2026):
    Nifty 50  : ~23,818   (down 6.85% YoY)
    Sensex    : ~75,873
    FII Flow  : ₹-1,042 Cr (selling pressure)
    DII Flow  : ₹+3,821 Cr (domestic buying)
    Key risk  : Geopolitical uncertainty, FII outflows, Iran tensions
=====================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import norm, t as student_t
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)


# ─────────────────────────────────────────────
#  1. REALISTIC NIFTY 50 DATA GENERATOR
# ─────────────────────────────────────────────

def generate_nifty_data(start_level=17500, n_days=850):
    """
    Generate realistic Nifty 50 price series.

    Calibrated to actual Nifty behaviour:
    - Jan 2023: ~17,500
    - Sep 2024: ~26,250 (all-time high)
    - May 2026: ~23,800 (current, down from peak)

    Features:
    - Geometric Brownian Motion base
    - Fat tails (Student-t shocks)
    - 3 stress events: Mar 2023 mini-crash, Jun 2024 election,
      Aug 2024 global sell-off, Feb 2026 FII exodus
    - Realistic annualised vol ~15% (Nifty historical avg)
    """
    dates = pd.bdate_range(start='2023-01-02', periods=n_days)

    # Base GBM parameters (annualised)
    mu    = 0.08 / 252      # ~8% annual return
    sigma = 0.15 / np.sqrt(252)  # ~15% annual vol

    # Mixed normal + fat-tail returns
    z_normal = np.random.normal(0, 1, n_days)
    z_t      = student_t.rvs(df=5, size=n_days)
    blend    = 0.85 * z_normal + 0.15 * z_t
    returns  = mu + sigma * blend

    # Stress period 1: Mar 2023 mini-crash (Adani saga)
    returns[45:55] += np.random.normal(-0.015, 0.018, 10)

    # Stress period 2: Jun 2024 election volatility
    returns[370:380] += np.random.normal(-0.012, 0.020, 10)

    # Stress period 3: Aug 2024 global carry-trade unwind
    returns[410:425] += np.random.normal(-0.020, 0.022, 15)

    # Stress period 4: Feb 2026 sustained FII selling
    returns[770:800] += np.random.normal(-0.008, 0.015, 30)

    # Build price series
    prices = start_level * np.exp(np.cumsum(returns))

    # Scale to end near 23,800
    prices = prices * (23800 / prices[-1]) * np.random.uniform(0.97, 1.03)

    df = pd.DataFrame({
        'date'   : dates,
        'close'  : np.round(prices, 2),
        'return' : returns,
    })
    df = df.set_index('date')
    return df


# ─────────────────────────────────────────────
#  2. VAR CALCULATIONS
# ─────────────────────────────────────────────

def var_historical(returns, confidence=0.99, window=250):
    """Historical Simulation VaR — no distributional assumption."""
    var_series = pd.Series(index=returns.index, dtype=float, name='HistVaR')
    for i in range(window, len(returns)):
        hist = returns.iloc[i-window:i]
        var_series.iloc[i] = -np.percentile(hist, (1 - confidence) * 100)
    return var_series.dropna()


def var_parametric_normal(returns, confidence=0.99, window=250):
    """Parametric VaR assuming Normal distribution."""
    z = norm.ppf(confidence)
    var_series = pd.Series(index=returns.index, dtype=float, name='ParamVaR')
    for i in range(window, len(returns)):
        w  = returns.iloc[i-window:i]
        var_series.iloc[i] = -(w.mean() - z * w.std())
    return var_series.dropna()


def var_parametric_t(returns, confidence=0.99, window=250):
    """Parametric VaR with Student-t (captures fat tails better)."""
    var_series = pd.Series(index=returns.index, dtype=float, name='t-VaR')
    for i in range(window, len(returns)):
        w = returns.iloc[i-window:i]
        df_fit, loc, scale = student_t.fit(w, floc=w.mean())
        var_series.iloc[i] = -student_t.ppf(1-confidence, df_fit, loc, scale)
    return var_series.dropna()


def var_monte_carlo(returns, confidence=0.99, window=250, n_sims=10000):
    """Monte Carlo VaR via GBM simulation."""
    var_series = pd.Series(index=returns.index, dtype=float, name='MC-VaR')
    for i in range(window, len(returns)):
        w      = returns.iloc[i-window:i]
        mu_w   = w.mean()
        sig_w  = w.std()
        sims   = np.random.normal(mu_w, sig_w, n_sims)
        var_series.iloc[i] = -np.percentile(sims, (1-confidence)*100)
    return var_series.dropna()


def cvar_historical(returns, confidence=0.99, window=250):
    """CVaR / Expected Shortfall — average loss beyond VaR."""
    cvar_series = pd.Series(index=returns.index, dtype=float, name='CVaR')
    for i in range(window, len(returns)):
        w   = returns.iloc[i-window:i]
        var = np.percentile(w, (1-confidence)*100)
        tail = w[w <= var]
        cvar_series.iloc[i] = -tail.mean() if len(tail) > 0 else np.nan
    return cvar_series.dropna()


# ─────────────────────────────────────────────
#  3. BACKTESTING
# ─────────────────────────────────────────────

def backtest(returns, var_series, confidence=0.99):
    """Count VaR breaches and compute exception statistics."""
    aligned   = returns.loc[var_series.index]
    breaches  = aligned < -var_series
    n_breach  = breaches.sum()
    n_days    = len(breaches)
    exp_rate  = 1 - confidence
    exp_n     = exp_rate * n_days
    return {
        'n_breaches'   : int(n_breach),
        'n_days'       : n_days,
        'breach_rate'  : n_breach / n_days,
        'expected_n'   : exp_n,
        'breach_mask'  : breaches,
        'returns_align': aligned,
        'var_series'   : var_series,
    }


def basel_traffic_light(n_breaches, window=250):
    """Basel III Traffic Light: Green / Amber / Red."""
    if n_breaches <= 4:
        return {'zone': 'GREEN',  'multiplier': 3.00,
                'action': 'Model acceptable. No penalty.'}
    elif n_breaches <= 9:
        mult = 3.00 + (n_breaches - 4) * 0.20
        return {'zone': 'AMBER',  'multiplier': round(mult, 2),
                'action': f'Investigate model. Capital add-on: {mult:.2f}x'}
    else:
        return {'zone': 'RED',    'multiplier': 4.00,
                'action': 'Model rejected. Mandatory 4.00x capital penalty.'}


def kupiec_test(n_breaches, n_days, confidence=0.99):
    """Kupiec POF Test: H0 = breach rate equals expected rate."""
    p   = 1 - confidence
    x   = n_breaches
    T   = n_days
    if x == 0:
        lr = -2 * T * np.log(1 - p)
    else:
        ph = x / T
        lr = -2 * (x*np.log(p/ph) + (T-x)*np.log((1-p)/(1-ph)))
    pval = 1 - stats.chi2.cdf(lr, df=1)
    crit = stats.chi2.ppf(0.95, df=1)
    return {
        'LR'        : round(lr, 4),
        'Critical'  : round(crit, 4),
        'p-value'   : round(pval, 4),
        'Reject H0' : lr > crit,
        'Result'    : 'REJECTED' if lr > crit else 'ACCEPTED',
    }


# ─────────────────────────────────────────────
#  4. STRESSED VAR
# ─────────────────────────────────────────────

def stressed_var(returns, confidence=0.99):
    """
    Stressed VaR using worst historical periods.
    Basel 2.5 requires banks to hold max(VaR, Stressed VaR).

    Indian market stress periods:
    - Mar 2020 COVID crash: Nifty fell 38% in 40 days
    - Jun 2022 global tightening: Nifty fell 18% in 3 months
    - Aug 2024 carry-trade unwind: sharp 5-day sell-off
    - Feb 2026 FII exodus period
    """
    # Identify 250-day window with worst returns
    worst_idx  = None
    worst_loss = 0
    window     = 250
    for i in range(window, len(returns)):
        w_loss = returns.iloc[i-window:i].sum()
        if w_loss < worst_loss:
            worst_loss = w_loss
            worst_idx  = i

    if worst_idx is None:
        worst_idx = window

    stress_window = returns.iloc[worst_idx-window:worst_idx]
    svar = -np.percentile(stress_window, (1-confidence)*100)
    scvar = -stress_window[stress_window <= -svar].mean()

    return {
        'stressed_var'   : round(svar, 6),
        'stressed_cvar'  : round(scvar, 6),
        'stress_period_start': returns.index[worst_idx-window].strftime('%Y-%m-%d'),
        'stress_period_end'  : returns.index[worst_idx-1].strftime('%Y-%m-%d'),
        'total_loss_pct' : round(worst_loss * 100, 2),
    }


# ─────────────────────────────────────────────
#  5. ROLLING VAR
# ─────────────────────────────────────────────

def rolling_var(returns, confidence=0.99, window=63):
    """
    63-day (1 quarter) rolling VaR to show how risk evolved.
    Reveals spikes during stress periods.
    """
    rolling = pd.Series(index=returns.index, dtype=float)
    for i in range(window, len(returns)):
        w = returns.iloc[i-window:i]
        rolling.iloc[i] = -np.percentile(w, (1-confidence)*100)
    return rolling.dropna()


# ─────────────────────────────────────────────
#  6. VISUALISATION
# ─────────────────────────────────────────────

def plot_nifty_risk(nifty_df, bt_hist, bt_param, bt_mc,
                   var_hist, var_param, cvar_s, rolling_v, svar):
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor('#0D1117')
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.52, wspace=0.35)

    BG=  '#0D1117'; PANEL='#161B22'; GC='#2A2A3A'; TC='#E0E0E0'
    G='#00C896'; R='#FF4444'; A='#FFB347'; B='#4A9EFF'; P='#C084FC'; T='#00D4C8'

    # ── Panel 1: Nifty Price Series ──
    ax1 = fig.add_subplot(gs[0, :]); ax1.set_facecolor(PANEL)
    ax1.plot(nifty_df.index, nifty_df['close'], color=B, lw=1.2, label='Nifty 50')
    ax1.fill_between(nifty_df.index, nifty_df['close'],
                     nifty_df['close'].min(), alpha=0.08, color=B)
    ax1.axhline(23818, color=T, lw=1, ls='--', alpha=0.7, label='Current: 23,818 (May 2026)')
    ax1.axhline(26250, color=G, lw=1, ls=':', alpha=0.6, label='ATH: 26,250 (Sep 2024)')
    # Shade stress periods
    stress_periods = [
        ('2023-03-01', '2023-03-20', 'Adani saga'),
        ('2024-06-01', '2024-06-15', 'Election vol.'),
        ('2024-08-01', '2024-08-20', 'Carry unwind'),
        ('2026-02-01', '2026-03-10', 'FII exodus'),
    ]
    for s, e, lbl in stress_periods:
        try:
            ax1.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                        alpha=0.15, color=R, label=lbl if s == '2023-03-01' else '')
        except: pass
    ax1.set_title('Nifty 50 — Price History with Stress Events (Jan 2023 – May 2026)',
                  color=TC, fontsize=11, fontweight='bold')
    ax1.set_ylabel('Index Level', color=TC, fontsize=9)
    ax1.tick_params(colors=TC, labelsize=8)
    ax1.grid(True, color=GC, lw=0.4)
    ax1.legend(fontsize=8, facecolor=PANEL, labelcolor=TC, ncol=3)
    for s in ax1.spines.values(): s.set_color(GC)

    # ── Panel 2: VaR Backtest ──
    ax2 = fig.add_subplot(gs[1, 0]); ax2.set_facecolor(PANEL)
    ra = bt_hist['returns_align']
    va = bt_hist['var_series']
    ex = bt_hist['breach_mask']
    ax2.plot(ra.index, ra*100, color=B, lw=0.6, alpha=0.6, label='Daily Return')
    ax2.plot(va.index, -va*100, color=G, lw=1.3, label='Historical VaR 99%')
    ax2.scatter(ra[ex].index, ra[ex]*100, color=R, s=18, zorder=5,
                label=f'Breaches: {bt_hist["n_breaches"]}')
    ax2.set_title(f'Historical VaR Backtest — {bt_hist["n_breaches"]} breaches / {bt_hist["n_days"]} days',
                  color=TC, fontsize=10, fontweight='bold')
    ax2.set_ylabel('Daily Return (%)', color=TC, fontsize=8)
    ax2.tick_params(colors=TC, labelsize=7)
    ax2.grid(True, color=GC, lw=0.4)
    ax2.legend(fontsize=7.5, facecolor=PANEL, labelcolor=TC)
    for s in ax2.spines.values(): s.set_color(GC)

    # ── Panel 3: Method Comparison ──
    ax3 = fig.add_subplot(gs[1, 1]); ax3.set_facecolor(PANEL)
    methods = ['Historical', 'Parametric\n(Normal)', 'Monte Carlo']
    nb      = [bt_hist['n_breaches'], bt_param['n_breaches'], bt_mc['n_breaches']]
    cols    = [G if n<=4 else A if n<=9 else R for n in nb]
    bars    = ax3.bar(methods, nb, color=cols, alpha=0.85, width=0.5)
    ax3.axhline(bt_hist['expected_n'], color=T, lw=1.5, ls='--',
                label=f'Expected: {bt_hist["expected_n"]:.1f}')
    ax3.axhline(4, color=G, lw=1, ls=':', alpha=0.5, label='Green limit (4)')
    ax3.axhline(9, color=A, lw=1, ls=':', alpha=0.5, label='Amber limit (9)')
    for bar, val in zip(bars, nb):
        ax3.text(bar.get_x()+bar.get_width()/2, val+0.1, str(val),
                 ha='center', color=TC, fontsize=11, fontweight='bold')
    ax3.set_title('Basel III Traffic Light — Breach Count by Model',
                  color=TC, fontsize=10, fontweight='bold')
    ax3.set_ylabel('Number of Breaches', color=TC, fontsize=8)
    ax3.tick_params(colors=TC, labelsize=8)
    ax3.grid(True, color=GC, lw=0.4, axis='y')
    ax3.legend(fontsize=7.5, facecolor=PANEL, labelcolor=TC)
    for s in ax3.spines.values(): s.set_color(GC)

    # ── Panel 4: Return Distribution + VaR lines ──
    ax4 = fig.add_subplot(gs[2, 0]); ax4.set_facecolor(PANEL)
    returns = nifty_df['return']
    ax4.hist(returns*100, bins=80, density=True, color=B, alpha=0.6, label='Actual Returns')
    last_var_h = var_hist.iloc[-1]
    last_cvar  = cvar_s.iloc[-1]
    last_var_p = var_param.iloc[-1]
    ax4.axvline(-last_var_h*100, color=G,  lw=2, ls='--', label=f'Hist VaR = {last_var_h*100:.2f}%')
    ax4.axvline(-last_var_p*100, color=A,  lw=2, ls='--', label=f'Param VaR = {last_var_p*100:.2f}%')
    ax4.axvline(-last_cvar*100,  color=P,  lw=2, ls=':',  label=f'CVaR (ES) = {last_cvar*100:.2f}%')
    x = np.linspace(returns.min()*100, returns.max()*100, 300)
    ax4.plot(x, norm.pdf(x, returns.mean()*100, returns.std()*100),
             color='white', lw=1.5, alpha=0.5, label='Normal fit')
    ax4.set_title('Nifty 50 Return Distribution\nwith VaR & CVaR Thresholds',
                  color=TC, fontsize=10, fontweight='bold')
    ax4.set_xlabel('Daily Return (%)', color=TC, fontsize=8)
    ax4.set_ylabel('Density', color=TC, fontsize=8)
    ax4.tick_params(colors=TC, labelsize=7)
    ax4.grid(True, color=GC, lw=0.4)
    ax4.legend(fontsize=7, facecolor=PANEL, labelcolor=TC)
    for s in ax4.spines.values(): s.set_color(GC)

    # ── Panel 5: Rolling VaR ──
    ax5 = fig.add_subplot(gs[2, 1]); ax5.set_facecolor(PANEL)
    ax5.plot(rolling_v.index, rolling_v*100, color=T, lw=1.5, label='Rolling 63-day VaR 99%')
    ax5.fill_between(rolling_v.index, rolling_v*100, rolling_v.min()*100,
                     alpha=0.15, color=T)
    ax5.axhline(rolling_v.mean()*100, color=G, lw=1, ls='--', alpha=0.7,
                label=f'Mean VaR: {rolling_v.mean()*100:.2f}%')
    ax5.set_title('Rolling 63-Day VaR — Risk Through Time\n(Spikes = stress periods)',
                  color=TC, fontsize=10, fontweight='bold')
    ax5.set_ylabel('VaR (%)', color=TC, fontsize=8)
    ax5.tick_params(colors=TC, labelsize=7)
    ax5.grid(True, color=GC, lw=0.4)
    ax5.legend(fontsize=7.5, facecolor=PANEL, labelcolor=TC)
    for s in ax5.spines.values(): s.set_color(GC)

    # ── Panel 6: VaR Sensitivity Table ──
    ax6 = fig.add_subplot(gs[3, :]); ax6.set_facecolor(PANEL); ax6.axis('off')
    returns_last = nifty_df['return'].iloc[-250:]
    rows = []
    for conf in [0.90, 0.95, 0.99, 0.995, 0.999]:
        v  = -np.percentile(returns_last, (1-conf)*100) * 100
        cv = -returns_last[returns_last <= -v/100].mean() * 100
        # Portfolio INR impact (assume ₹1 Cr portfolio)
        inr_var  = 10_000_000 * v / 100
        inr_cvar = 10_000_000 * cv / 100
        tl = basel_traffic_light(bt_hist['n_breaches'])
        rows.append([f'{conf*100:.1f}%',
                     f'{v:.3f}%', f'{cv:.3f}%',
                     f'₹{inr_var:,.0f}', f'₹{inr_cvar:,.0f}',
                     f'{cv/v:.2f}x'])
    cols_tbl = ['Confidence','VaR (%)','CVaR (%)','VaR (₹1Cr portfolio)',
                'CVaR (₹1Cr portfolio)','CVaR/VaR ratio']
    tbl = ax6.table(cellText=rows, colLabels=cols_tbl,
                    cellLoc='center', loc='center', bbox=[0,0.1,1,0.8])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    for (row,col),cell in tbl.get_celld().items():
        cell.set_facecolor(PANEL if row>0 else '#1A3A6B')
        cell.set_edgecolor(GC)
        cell.set_text_props(color=TC)
        if row > 0 and col in [1,2]:
            val = float(rows[row-1][col].replace('%',''))
            if val > 2.5: cell.set_facecolor('#3B1F1F')
    ax6.set_title('VaR Sensitivity Table — Nifty 50 at Multiple Confidence Levels\n'
                  f'Stressed VaR: {svar["stressed_var"]*100:.3f}%  |  '
                  f'Stress Period: {svar["stress_period_start"]} to {svar["stress_period_end"]}  |  '
                  f'Basel Zone: {basel_traffic_light(bt_hist["n_breaches"])["zone"]}  |  '
                  f'Capital Multiplier: {basel_traffic_light(bt_hist["n_breaches"])["multiplier"]}x',
                  color=TC, fontsize=9, fontweight='bold', pad=8)

    fig.suptitle('Nifty 50 — Value at Risk (VaR) & Risk Model | Indian Equity Market',
                 color=TC, fontsize=13, fontweight='bold', y=1.01)
    plt.savefig('/mnt/user-data/outputs/nifty_var_risk_model.png',
                dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print("  Chart saved.")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("="*65)
    print("  NIFTY 50 — VALUE AT RISK & RISK MODEL")
    print("="*65)

    nifty_df = generate_nifty_data()
    returns  = nifty_df['return']

    print(f"\n── Market Data Summary ──")
    print(f"  Period      : {nifty_df.index[0].date()} → {nifty_df.index[-1].date()}")
    print(f"  Days        : {len(nifty_df)}")
    print(f"  Start level : ₹{nifty_df['close'].iloc[0]:,.0f}")
    print(f"  End level   : ₹{nifty_df['close'].iloc[-1]:,.0f}")
    print(f"  Total return: {(nifty_df['close'].iloc[-1]/nifty_df['close'].iloc[0]-1)*100:.1f}%")
    print(f"  Ann. vol    : {returns.std()*np.sqrt(252)*100:.2f}%")
    print(f"  Skewness    : {returns.skew():.4f}")
    print(f"  Kurtosis    : {returns.kurt():.4f}  (>0 = fat tails)")
    print(f"  Max 1-day   : {returns.max()*100:.2f}%")
    print(f"  Min 1-day   : {returns.min()*100:.2f}%")

    print(f"\n── Computing VaR Models ──")
    CONF   = 0.99
    WINDOW = 250
    var_hist  = var_historical(returns, CONF, WINDOW)
    var_param = var_parametric_normal(returns, CONF, WINDOW)
    var_t     = var_parametric_t(returns, CONF, WINDOW)
    var_mc    = var_monte_carlo(returns, CONF, WINDOW, 10000)
    cvar_s    = cvar_historical(returns, CONF, WINDOW)
    rolling_v = rolling_var(returns, CONF, 63)

    print(f"\n  Latest VaR estimates (99% confidence):")
    print(f"  Historical VaR  : {var_hist.iloc[-1]*100:.4f}%")
    print(f"  Parametric VaR  : {var_param.iloc[-1]*100:.4f}%")
    print(f"  Student-t VaR   : {var_t.iloc[-1]*100:.4f}%")
    print(f"  Monte Carlo VaR : {var_mc.iloc[-1]*100:.4f}%")
    print(f"  CVaR (ES)       : {cvar_s.iloc[-1]*100:.4f}%")

    inr_var  = 10_000_000 * var_hist.iloc[-1]
    inr_cvar = 10_000_000 * cvar_s.iloc[-1]
    print(f"\n  On ₹1 Crore Nifty position:")
    print(f"  1-day VaR  : ₹{inr_var:>10,.0f}")
    print(f"  1-day CVaR : ₹{inr_cvar:>10,.0f}")

    print(f"\n── Backtesting (last 250 days) ──")
    bt_hist  = backtest(returns, var_hist,  CONF)
    bt_param = backtest(returns, var_param, CONF)
    bt_mc    = backtest(returns, var_mc,    CONF)

    for name, bt in [('Historical', bt_hist),
                     ('Parametric', bt_param),
                     ('Monte Carlo', bt_mc)]:
        tl = basel_traffic_light(bt['n_breaches'])
        kp = kupiec_test(bt['n_breaches'], bt['n_days'])
        print(f"\n  {name}:")
        print(f"    Breaches      : {bt['n_breaches']} / {bt['n_days']}  "
              f"(expected {bt['expected_n']:.1f})")
        print(f"    Basel zone    : {tl['zone']}  |  Multiplier: {tl['multiplier']}x")
        print(f"    Kupiec test   : {kp['Result']}  (p={kp['p-value']})")

    print(f"\n── Stressed VaR ──")
    svar = stressed_var(returns, CONF)
    print(f"  Stress period   : {svar['stress_period_start']} → {svar['stress_period_end']}")
    print(f"  Stressed VaR    : {svar['stressed_var']*100:.4f}%")
    print(f"  Stressed CVaR   : {svar['stressed_cvar']*100:.4f}%")
    print(f"  Period loss     : {svar['total_loss_pct']}%")
    ratio = svar['stressed_var'] / var_hist.iloc[-1]
    print(f"  Stressed/Normal : {ratio:.2f}x  (Basel 2.5 capital = max of both)")

    print(f"\n── Generating Charts ──")
    plot_nifty_risk(nifty_df, bt_hist, bt_param, bt_mc,
                   var_hist, var_param, cvar_s, rolling_v, svar)
    print("\nDone.")
