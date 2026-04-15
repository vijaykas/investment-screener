#!/usr/bin/env python3
"""
============================================================
  INVESTMENT INTELLIGENCE — Unified Runner
  Author: Claude (Vijay's Investment Project)
  Updated: 2026-04-12
============================================================

USAGE:
  python3 invest.py                    # Interactive menu
  python3 invest.py --all              # Full pipeline, all phases
  python3 invest.py --phase 1 2 6      # Run specific phases
  python3 invest.py --screener         # Phase 1 only
  python3 invest.py --optimizer        # Phase 2 only
  python3 invest.py --backtest         # Phase 3 only
  python3 invest.py --monitor          # Phase 4 only
  python3 invest.py --etf              # Phase 4b only (ETF screener)
  python3 invest.py --ml               # Phase 5 only
  python3 invest.py --sentiment        # Phase 6 only
  python3 invest.py --status           # Show last run times + output files

MAKE EXECUTABLE (one-time, run in terminal):
  chmod +x invest.py
  ./invest.py --all
============================================================
"""

import argparse
import importlib.util
import os
import sys
import time
import json
import traceback
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────
#  TERMINAL COLOURS (works on macOS / Linux)
# ─────────────────────────────────────────────

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    GREEN  = "\033[92m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    PURPLE = "\033[95m"
    WHITE  = "\033[97m"
    GREY   = "\033[90m"
    BG_DARK= "\033[40m"

def cprint(text, color=C.WHITE, bold=False):
    prefix = C.BOLD if bold else ""
    print(f"{prefix}{color}{text}{C.RESET}")

def bar(n, total=40, fill="█", empty="░"):
    filled = int(n / 100 * total)
    return fill * filled + empty * (total - filled)

# ─────────────────────────────────────────────
#  PHASE REGISTRY
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).parent

PHASES = {
    1: {
        "name":    "Stock Screener",
        "emoji":   "🔍",
        "file":    "stock_screener.py",
        "fn":      "run_screener",
        "desc":    "Scan 55 stocks — technical + fundamental scoring (0–75)",
        "outputs": ["data/stock_screener_results.csv", "charts/"],
        "deps":    [],
        "flag":    "screener",
        "color":   C.GREEN,
    },
    "4b": {
        "name":    "ETF Screener",
        "emoji":   "📈",
        "file":    "etf_screener.py",
        "fn":      "run_etf_screener",
        "desc":    "Score 34 ETFs — Technical + Momentum vs SPY + Expense/Quality (0–100)",
        "outputs": ["data/etf_screener_results.csv"],
        "deps":    [],
        "flag":    "etf",
        "color":   C.GREEN,
    },
    "4c": {
        "name":    "Crypto Screener",
        "emoji":   "₿",
        "file":    "crypto_screener.py",
        "fn":      "run_crypto_screener",
        "desc":    "Score 30 quality cryptos — Technical + Momentum vs BTC + On-chain Quality + Sentiment",
        "outputs": ["data/crypto_screener_results.csv", "data/crypto_cycle.json"],
        "deps":    [],
        "flag":    "crypto",
        "color":   C.YELLOW,
    },
    2: {
        "name":    "Portfolio Optimizer",
        "emoji":   "📐",
        "file":    "portfolio_optimizer.py",
        "fn":      "run_optimizer",
        "desc":    "MPT optimisation — 15k Monte Carlo, Sharpe/MinVol/MaxReturn",
        "outputs": ["data/portfolio_results.csv", "charts/efficient_frontier.html"],
        "deps":    [1],
        "flag":    "optimizer",
        "color":   C.BLUE,
    },
    3: {
        "name":    "Backtester",
        "emoji":   "📊",
        "file":    "backtester.py",
        "fn":      "run_backtest",
        "desc":    "5 strategies × walk-forward validation vs SPY benchmark",
        "outputs": ["data/backtest_results.csv", "charts/equity_curves.html"],
        "deps":    [],
        "flag":    "backtest",
        "color":   C.YELLOW,
    },
    4: {
        "name":    "Daily Monitor",
        "emoji":   "📡",
        "file":    "daily_monitor.py",
        "fn":      "run_monitor",
        "desc":    "Morning signal scan — changes, alerts, P&L, HTML report",
        "outputs": ["daily_reports/", "data/signal_history.json"],
        "deps":    [],
        "flag":    "monitor",
        "color":   C.CYAN,
    },
    5: {
        "name":    "ML Predictor",
        "emoji":   "🤖",
        "file":    "ml_predictor.py",
        "fn":      "run_ml_predictor",
        "desc":    "RF + XGBoost + LR ensemble — 45+ features, 12 horizons (1m–12m)",
        "outputs": ["data/ml_predictions.csv", "data/ml_model_performance.csv", "charts/ml_prediction_timeline.html"],
        "deps":    [],
        "flag":    "ml",
        "color":   C.PURPLE,
    },
    6: {
        "name":    "Sentiment Analyzer",
        "emoji":   "📰",
        "file":    "sentiment_analyzer.py",
        "fn":      "run_sentiment_analysis",
        "desc":    "News NLP + Analyst consensus + Earnings surprise + Insider activity",
        "outputs": ["data/sentiment_results.csv", "charts/sentiment_dashboard.html"],
        "deps":    [1],
        "flag":    "sentiment",
        "color":   C.GREEN,
    },
    7: {
        "name":    "News & Sentiment Engine",
        "emoji":   "📡",
        "file":    "news_sentiment.py",
        "fn":      "run_news_sentiment",
        "desc":    "Live news + sentiment for all stocks + ETFs — AV / Finnhub / FMP / synthetic fallback",
        "outputs": ["data/news_sentiment.csv", "data/macro_themes.json"],
        "deps":    [],
        "flag":    "news",
        "color":   C.CYAN,
    },
    8: {
        "name":    "Top 20 Predictions",
        "emoji":   "🏆",
        "file":    "top20_picker.py",
        "fn":      "run_top20",
        "desc":    "Rank top 20 stocks + ETFs by Yield Potential Score (Technical + ML/Momentum + Sentiment)",
        "outputs": ["data/top20_predictions.csv"],
        "deps":    [1, "4b", 7],
        "flag":    "top20",
        "color":   C.YELLOW,
    },
}

STATE_FILE = BASE_DIR / ".invest_state.json"

# ─────────────────────────────────────────────
#  STATE TRACKING
# ─────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}

def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def record_run(phase_num: int, success: bool, elapsed: float, state: dict):
    state[str(phase_num)] = {
        "last_run":  datetime.now().strftime("%Y-%m-%d %H:%M"),
        "success":   success,
        "elapsed_s": round(elapsed, 1),
    }
    save_state(state)

# ─────────────────────────────────────────────
#  MODULE LOADER
# ─────────────────────────────────────────────

def load_phase_module(phase_key):
    """Dynamically import a phase script and return the module."""
    info = PHASES[phase_key]
    path = BASE_DIR / info["file"]
    if not path.exists():
        raise FileNotFoundError(f"{info['file']} not found in {BASE_DIR}")
    mod_name = f"phase_{str(phase_key).replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ─────────────────────────────────────────────
#  BANNER & STATUS
# ─────────────────────────────────────────────

def print_banner():
    state = load_state()
    cprint("\n" + "═" * 62, C.CYAN)
    cprint("  💹  INVESTMENT INTELLIGENCE STACK", C.CYAN, bold=True)
    cprint(f"  {datetime.now().strftime('%A, %B %d, %Y  %H:%M')}", C.GREY)
    cprint("═" * 62, C.CYAN)

    for num, info in PHASES.items():
        s = state.get(str(num), {})
        if s:
            ts      = s.get("last_run", "—")
            ok      = s.get("success", False)
            elapsed = s.get("elapsed_s", 0)
            status  = f"{C.GREEN}✅ {ts}  ({elapsed}s){C.RESET}" if ok else f"{C.RED}❌ failed{C.RESET}"
        else:
            status = f"{C.GREY}— not run yet{C.RESET}"

        cprint(f"  {info['emoji']}  Phase {num:<3}: {info['name']:<22}", info["color"], bold=False)
        print(f"     {status}")

    cprint("═" * 62 + "\n", C.CYAN)

def print_phase_header(phase_num: int):
    info = PHASES[phase_num]
    cprint(f"\n{'─' * 62}", info["color"])
    cprint(f"  {info['emoji']}  PHASE {phase_num}: {info['name'].upper()}", info["color"], bold=True)
    cprint(f"  {info['desc']}", C.GREY)
    cprint(f"{'─' * 62}", info["color"])

def print_outputs(phase_num: int):
    info = PHASES[phase_num]
    cprint(f"\n  📁 Outputs:", C.GREY)
    for out in info["outputs"]:
        full = BASE_DIR / out
        exists = full.exists()
        sym    = "✅" if exists else "—"
        cprint(f"     {sym}  {out}", C.GREY if not exists else C.WHITE)

# ─────────────────────────────────────────────
#  DEPENDENCY CHECK
# ─────────────────────────────────────────────

def check_deps(phase_num: int, state: dict) -> bool:
    deps = PHASES[phase_num]["deps"]
    missing = []
    for dep in deps:
        dep_state = state.get(str(dep), {})
        if not dep_state.get("success"):
            # Also check if output file exists as fallback
            dep_outputs = PHASES[dep]["outputs"]
            file_exists = any((BASE_DIR / o).exists() for o in dep_outputs)
            if not file_exists:
                missing.append(dep)

    if missing:
        dep_names = ", ".join(f"Phase {d} ({PHASES[d]['name']})" for d in missing)
        cprint(f"\n  ⚠️  Phase {phase_num} works best after: {dep_names}", C.YELLOW)
        ans = input(f"  Continue anyway? [y/N]: ").strip().lower()
        return ans == "y"
    return True

# ─────────────────────────────────────────────
#  PHASE RUNNER
# ─────────────────────────────────────────────

def run_phase(phase_key, state: dict) -> tuple[bool, float]:
    """Load and execute a single phase. Returns (success, elapsed_seconds)."""
    info = PHASES[phase_key]
    print_phase_header(phase_key)

    # Check dependencies
    if not check_deps(phase_key, state):
        return False, 0.0

    t0 = time.time()
    success = False

    try:
        mod = load_phase_module(phase_key)
        fn  = getattr(mod, info["fn"])
        fn()
        elapsed = time.time() - t0
        success = True
        cprint(f"\n  ✅ Phase {phase_key} complete in {elapsed:.1f}s", C.GREEN, bold=True)
        print_outputs(phase_key)
    except KeyboardInterrupt:
        elapsed = time.time() - t0
        cprint(f"\n  ⏹  Phase {phase_key} interrupted after {elapsed:.1f}s", C.YELLOW)
        raise
    except Exception as e:
        elapsed = time.time() - t0
        cprint(f"\n  ❌ Phase {phase_key} failed after {elapsed:.1f}s", C.RED, bold=True)
        cprint(f"  Error: {e}", C.RED)
        if "--debug" in sys.argv:
            traceback.print_exc()
        else:
            cprint(f"  (Run with --debug for full traceback)", C.GREY)

    record_run(str(phase_key), success, elapsed, state)
    return success, elapsed

# ─────────────────────────────────────────────
#  PIPELINE RUNNER
# ─────────────────────────────────────────────

def run_pipeline(phase_nums: list[int], state: dict):
    """Run multiple phases in sequence and print a final summary."""
    total_start = time.time()
    results = {}

    cprint(f"\n  🚀 Running {len(phase_nums)} phase(s): "
           f"{', '.join(str(n) for n in phase_nums)}\n", C.WHITE, bold=True)

    for phase_num in phase_nums:
        try:
            success, elapsed = run_phase(phase_num, state)
            results[phase_num] = (success, elapsed)
        except KeyboardInterrupt:
            cprint("\n\n  ⏹  Pipeline interrupted by user.", C.YELLOW)
            break

    # Summary
    total_elapsed = time.time() - total_start
    cprint(f"\n{'═' * 62}", C.CYAN)
    cprint(f"  PIPELINE SUMMARY  ({total_elapsed:.0f}s total)", C.CYAN, bold=True)
    cprint(f"{'═' * 62}", C.CYAN)

    for num, (ok, t) in results.items():
        info = PHASES[num]
        sym  = f"{C.GREEN}✅{C.RESET}" if ok else f"{C.RED}❌{C.RESET}"
        cprint(f"  {sym}  Phase {num}: {info['name']:<22}  {t:.1f}s", info["color"])

    n_ok   = sum(1 for ok, _ in results.values() if ok)
    n_fail = len(results) - n_ok

    if n_fail == 0:
        cprint(f"\n  🎉 All {n_ok} phases completed successfully!", C.GREEN, bold=True)
    else:
        cprint(f"\n  ⚠️  {n_ok} passed, {n_fail} failed.", C.YELLOW, bold=True)

    # Show key output files
    cprint(f"\n  📁 Key Output Files:", C.GREY)
    key_files = [
        ("data/stock_screener_results.csv",    "Screener rankings"),
        ("data/etf_screener_results.csv",      "ETF scores & momentum"),
        ("data/crypto_screener_results.csv",   "Crypto scores (Phase 4c)"),
        ("data/crypto_cycle.json",             "BTC market cycle data"),
        ("data/portfolio_results.csv",         "Portfolio allocations"),
        ("data/backtest_results.csv",          "Strategy performance"),
        ("data/ml_predictions.csv",            "ML predictions (1m–12m)"),
        ("data/sentiment_results.csv",         "Sentiment scores"),
        ("data/news_sentiment.csv",            "News sentiment (Phase 7)"),
        ("data/macro_themes.json",             "Macro market themes"),
        ("data/top20_predictions.csv",         "Top 20 high-yield predictions"),
        (f"daily_reports/{datetime.now().strftime('%Y-%m-%d')}.html", "Today's report"),
        ("charts/efficient_frontier.html",     "Efficient Frontier chart"),
        ("charts/ml_prediction_timeline.html", "ML 12-month timeline"),
        ("charts/sentiment_dashboard.html",    "Sentiment dashboard"),
    ]
    for fname, label in key_files:
        path = BASE_DIR / fname
        if path.exists():
            cprint(f"     ✅  {fname:<48} {C.GREY}{label}{C.RESET}", C.WHITE)

    cprint(f"\n{'═' * 62}\n", C.CYAN)
    cprint("  ⚠️  For research purposes only — not financial advice.", C.GREY)
    cprint("  Always validate signals and consult a financial advisor.\n", C.GREY)

# ─────────────────────────────────────────────
#  INTERACTIVE MENU
# ─────────────────────────────────────────────

def interactive_menu(state: dict):
    """Show an interactive menu when no CLI args are provided."""
    while True:
        print_banner()
        cprint("  SELECT AN OPTION:", C.WHITE, bold=True)
        print()

        for num, info in PHASES.items():
            s       = state.get(str(num), {})
            last    = s.get("last_run", "never")
            ok_sym  = f"{C.GREEN}✓{C.RESET}" if s.get("success") else f"{C.GREY}○{C.RESET}"
            cprint(f"  {ok_sym} [{num}]  {info['emoji']}  {info['name']:<22}  "
                   f"{C.GREY}(last: {last}){C.RESET}", info["color"])

        print()
        cprint(f"  [A]  🚀  Run FULL PIPELINE (all phases in order)", C.WHITE, bold=True)
        cprint(f"  [Q]  Quick run: Screener → ETF → Sentiment → Optimizer", C.WHITE)
        cprint(f"  [S]  Status & output files", C.GREY)
        cprint(f"  [X]  Exit\n", C.GREY)

        choice = input(f"  {C.CYAN}Enter choice:{C.RESET} ").strip().upper()

        if choice == "X":
            cprint("\n  👋  Goodbye.\n", C.GREY)
            break

        elif choice == "A":
            run_pipeline(list(PHASES.keys()), state)
            input(f"\n  {C.GREY}Press Enter to return to menu...{C.RESET}")

        elif choice == "Q":
            run_pipeline([1, "4b", 7, 8, 4], state)
            input(f"\n  {C.GREY}Press Enter to return to menu...{C.RESET}")

        elif choice == "S":
            print_status(state)
            input(f"\n  {C.GREY}Press Enter to continue...{C.RESET}")

        elif choice.isdigit() and int(choice) in PHASES:
            run_pipeline([int(choice)], state)
            input(f"\n  {C.GREY}Press Enter to return to menu...{C.RESET}")

        else:
            # Support comma/space separated phase numbers e.g. "1 2 6" or "1,2,6"
            parts = choice.replace(",", " ").split()
            nums  = []
            for p in parts:
                if p.isdigit() and int(p) in PHASES:
                    nums.append(int(p))
            if nums:
                run_pipeline(nums, state)
                input(f"\n  {C.GREY}Press Enter to return to menu...{C.RESET}")
            else:
                cprint(f"\n  ⚠️  Invalid choice: '{choice}'\n", C.YELLOW)
                time.sleep(1)


def print_status(state: dict):
    cprint(f"\n{'─' * 62}", C.GREY)
    cprint(f"  STATUS & OUTPUT FILES", C.WHITE, bold=True)
    cprint(f"{'─' * 62}", C.GREY)

    for num, info in PHASES.items():
        s = state.get(str(num), {})
        cprint(f"\n  {info['emoji']}  Phase {num}: {info['name']}", info["color"], bold=True)
        if s:
            print(f"     Last run : {s.get('last_run','—')}")
            print(f"     Success  : {'✅ Yes' if s.get('success') else '❌ No'}")
            print(f"     Duration : {s.get('elapsed_s', '—')}s")
        else:
            cprint(f"     Not run yet", C.GREY)

        for out in info["outputs"]:
            p = BASE_DIR / out
            exists = "✅" if p.exists() else "❌"
            print(f"     {exists}  {out}")

    # Disk usage
    total_size = sum(
        f.stat().st_size for f in BASE_DIR.rglob("*")
        if f.is_file() and ".pkl" not in str(f)
    )
    cprint(f"\n  📦 Total output size (excl. caches): {total_size / 1024:.0f} KB", C.GREY)

# ─────────────────────────────────────────────
#  CLI ARGUMENT PARSER
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="invest.py",
        description="Investment Intelligence Stack — Unified Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 invest.py                   Interactive menu
  python3 invest.py --all             Full pipeline (all 6 phases)
  python3 invest.py --phase 1 6       Screener then Sentiment
  python3 invest.py --screener        Phase 1 only
  python3 invest.py --status          Show last run times
  python3 invest.py --phase 1 --debug Verbose error output
        """
    )
    p.add_argument("--all",       action="store_true",  help="Run all phases in sequence")
    p.add_argument("--phase",     nargs="+",            help="Run specific phases, e.g. --phase 1 4b 6",
                   metavar="N")
    p.add_argument("--screener",  action="store_true",  help="Phase 1: Stock Screener")
    p.add_argument("--optimizer", action="store_true",  help="Phase 2: Portfolio Optimizer")
    p.add_argument("--backtest",  action="store_true",  help="Phase 3: Backtester")
    p.add_argument("--monitor",   action="store_true",  help="Phase 4: Daily Monitor")
    p.add_argument("--etf",       action="store_true",  help="Phase 4b: ETF Screener")
    p.add_argument("--crypto",    action="store_true",  help="Phase 4c: Crypto Screener")
    p.add_argument("--ml",        action="store_true",  help="Phase 5: ML Predictor")
    p.add_argument("--sentiment", action="store_true",  help="Phase 6: Sentiment Analyzer")
    p.add_argument("--news",      action="store_true",  help="Phase 7: News & Sentiment Engine")
    p.add_argument("--top20",     action="store_true",  help="Phase 8: Top 20 High-Yield Predictions")
    p.add_argument("--quick",     action="store_true",  help="Quick run: Screener → ETF → Crypto → News → Top20 → Monitor")
    p.add_argument("--status",    action="store_true",  help="Show run status and output files")
    p.add_argument("--debug",     action="store_true",  help="Print full tracebacks on error")
    return p


def _all_phase_keys_ordered():
    """Return all phase keys in logical run order."""
    order = [1, 2, 3, 4, "4b", "4c", 5, 6, 7, 8]
    return [k for k in order if k in PHASES]


def resolve_phases(args) -> list:
    """Convert CLI flags to an ordered list of phase keys."""
    if args.all:
        return _all_phase_keys_ordered()
    if args.quick:
        return [1, "4b", "4c", 7, 8, 4]
    if args.phase:
        # Accept both int and string keys like "4b"
        resolved = []
        for p in args.phase:
            try:
                k = int(p)
            except ValueError:
                k = p   # e.g. "4b"
            if k in PHASES:
                resolved.append(k)
        return resolved

    # Individual flags
    phase_map = {
        "screener":  1,
        "optimizer": 2,
        "backtest":  3,
        "monitor":   4,
        "etf":       "4b",
        "crypto":    "4c",
        "ml":        5,
        "sentiment": 6,
        "news":      7,
        "top20":     8,
    }
    selected = [v for k, v in phase_map.items() if getattr(args, k, False)]
    # Keep logical order
    order = _all_phase_keys_ordered()
    return [k for k in order if k in selected]


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = build_parser()
    args   = parser.parse_args()
    state  = load_state()

    # Status-only mode
    if args.status:
        print_banner()
        print_status(state)
        return

    # If no action specified → interactive menu
    phases = resolve_phases(args)
    if not phases:
        try:
            interactive_menu(state)
        except KeyboardInterrupt:
            cprint("\n\n  👋  Goodbye.\n", C.GREY)
        return

    # Non-interactive: run the requested phases
    print_banner()
    run_pipeline(phases, state)


if __name__ == "__main__":
    main()
