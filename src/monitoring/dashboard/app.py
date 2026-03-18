"""
TraderBot Dashboard — Streamlit Web UI

Run with: streamlit run src/monitoring/dashboard/app.py

Pages:
1. Overview — Balance, PnL, open positions, controls
2. Trade History — Filterable trade log
3. Performance — Equity curve, drawdown, analytics
4. ML Status — Model info, feature importance, accuracy
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import streamlit as st

st.set_page_config(
    page_title="TraderBot",
    page_icon="\U0001f4b9",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.config import load_config
from src.monitoring.trade_journal import TradeJournal
from src.monitoring.performance import PerformanceTracker


@st.cache_resource
def get_config():
    return load_config()


@st.cache_resource
def get_journal():
    config = get_config()
    return TradeJournal(config)


@st.cache_resource
def get_performance():
    journal = get_journal()
    return PerformanceTracker(journal)


def main():
    config = get_config()
    journal = get_journal()
    perf = get_performance()

    # Sidebar
    st.sidebar.title("\U0001f4b9 TraderBot")
    st.sidebar.markdown(f"**Environment:** {config.broker_environment}")

    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Trade History", "Performance", "ML Status"],
    )

    if page == "Overview":
        show_overview(config, journal, perf)
    elif page == "Trade History":
        show_trade_history(journal)
    elif page == "Performance":
        show_performance(perf)
    elif page == "ML Status":
        show_ml_status(config)


def show_overview(config, journal, perf):
    """Overview page — key metrics and controls."""
    st.title("Overview")

    summary = perf.get_summary()

    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    starting = config.get("account.starting_balance", 500)
    balance = starting + summary["total_pnl"]

    col1.metric("Balance", f"${balance:.2f}", f"${summary['total_pnl']:+.2f}")
    col2.metric("Win Rate", f"{summary['win_rate']:.0%}", f"{summary['total_trades']} trades")
    col3.metric("Profit Factor", f"{summary['profit_factor']:.2f}")
    col4.metric("Max Drawdown", f"{summary['max_drawdown_pct']:.1f}%")

    st.divider()

    # Progress to target
    target = config.get("growth.target_balance", 6000)
    progress = min(balance / target, 1.0)
    st.subheader("Growth Progress")
    st.progress(progress, text=f"${balance:.0f} / ${target:.0f} ({progress:.0%})")

    st.divider()

    # Recent trades
    st.subheader("Recent Trades")
    trades_df = journal.get_trades(limit=10)
    if not trades_df.empty:
        display_cols = [
            "instrument", "direction", "entry_price", "exit_price",
            "pnl_pips", "pnl_zar", "ml_confidence", "exit_reason", "entry_time"
        ]
        available = [c for c in display_cols if c in trades_df.columns]
        st.dataframe(trades_df[available], use_container_width=True, hide_index=True)
    else:
        st.info("No trades recorded yet.")

    st.divider()

    # Quick stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Win", f"${summary['avg_win']:.2f}")
    col2.metric("Avg Loss", f"${summary['avg_loss']:.2f}")
    col3.metric("Sharpe Ratio", f"{summary['sharpe_ratio']:.2f}")


def show_trade_history(journal):
    """Trade history page — filterable trade log."""
    st.title("Trade History")

    # Filters
    col1, col2, col3 = st.columns(3)
    instrument = col1.selectbox(
        "Instrument", ["All", "EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"]
    )
    direction = col2.selectbox("Direction", ["All", "buy", "sell"])
    limit = col3.number_input("Show last N trades", value=50, min_value=10, max_value=500)

    trades_df = journal.get_trades(
        instrument=instrument if instrument != "All" else None,
        direction=direction if direction != "All" else None,
        limit=limit,
    )

    if trades_df.empty:
        st.info("No trades match the filter.")
        return

    # Summary for filtered trades
    if "pnl_zar" in trades_df.columns:
        pnl_col = trades_df["pnl_zar"].fillna(0)
        wins = (pnl_col > 0).sum()
        losses = (pnl_col <= 0).sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Trades", len(trades_df))
        col2.metric("Wins", int(wins))
        col3.metric("Losses", int(losses))
        col4.metric("Net PnL", f"${pnl_col.sum():.2f}")

    st.dataframe(trades_df, use_container_width=True, hide_index=True)


def show_performance(perf):
    """Performance page — charts and analytics."""
    st.title("Performance Analytics")

    summary = perf.get_summary()

    if summary["total_trades"] == 0:
        st.info("No completed trades to analyze.")
        return

    # Equity curve
    st.subheader("Equity Curve")
    equity = perf.get_equity_curve()
    if not equity.empty:
        st.line_chart(equity.set_index("time")["balance"])

    # Drawdown
    st.subheader("Drawdown")
    drawdown = perf.get_drawdown_series()
    if not drawdown.empty:
        st.area_chart(drawdown.set_index("time")["drawdown_pct"])

    st.divider()

    # Instrument breakdown
    st.subheader("Performance by Instrument")
    inst_df = perf.get_instrument_breakdown()
    if not inst_df.empty:
        st.dataframe(inst_df, use_container_width=True, hide_index=True)

    # Hourly breakdown
    st.subheader("Performance by Hour (UTC)")
    hourly_df = perf.get_hourly_breakdown()
    if not hourly_df.empty:
        st.bar_chart(hourly_df.set_index("hour")["total_pnl"])

    st.divider()

    # Exit reason breakdown
    st.subheader("Exit Reasons")
    exit_reasons = summary.get("exit_reasons", {})
    if exit_reasons:
        import pandas as pd
        er_df = pd.DataFrame(
            list(exit_reasons.items()), columns=["Reason", "Count"]
        )
        st.bar_chart(er_df.set_index("Reason"))


def show_ml_status(config):
    """ML Status page — model info and feature importance."""
    st.title("ML Model Status")

    from src.ml.trainer import MODEL_STORE

    # Load latest model metadata
    latest_path = MODEL_STORE / "latest_version.txt"
    if not latest_path.exists():
        st.info("No trained model found.")
        return

    version = latest_path.read_text().strip()
    meta_path = MODEL_STORE / f"model_{version}_meta.json"

    if not meta_path.exists():
        st.warning(f"Model {version} exists but metadata not found.")
        return

    import json
    with open(meta_path) as f:
        meta = json.load(f)

    # Model info
    col1, col2, col3 = st.columns(3)
    col1.metric("Version", meta.get("version", "?"))
    metrics = meta.get("metrics", {})
    col2.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
    col3.metric("AUC-ROC", f"{metrics.get('auc_roc', 0):.3f}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Precision", f"{metrics.get('precision', 0):.1%}")
    col2.metric("Recall", f"{metrics.get('recall', 0):.1%}")
    col3.metric("F1 Score", f"{metrics.get('f1', 0):.3f}")

    st.divider()

    # Training info
    st.subheader("Training Details")
    st.write(f"**Trained:** {meta.get('timestamp', '?')}")
    st.write(f"**Train size:** {meta.get('train_size', '?')} samples")
    st.write(f"**Test size:** {meta.get('test_size', '?')} samples")

    st.divider()

    # Feature importance
    st.subheader("Feature Importance")
    importance = meta.get("feature_importance", {})
    if importance:
        import pandas as pd
        imp_df = pd.DataFrame(
            sorted(importance.items(), key=lambda x: x[1], reverse=True),
            columns=["Feature", "Importance"],
        )
        st.bar_chart(imp_df.set_index("Feature"))

    st.divider()

    # Hyperparameters
    st.subheader("Hyperparameters")
    params = meta.get("xgb_params", {})
    if params:
        import pandas as pd
        params_df = pd.DataFrame(
            list(params.items()), columns=["Parameter", "Value"]
        )
        st.table(params_df)


if __name__ == "__main__":
    main()
