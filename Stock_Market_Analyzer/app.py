# app.py (corrected)
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import datetime

# Add src directory to path to import your modules (robust path handling)
# If your modules are in a folder named 'src' next to this script
src_path = Path(__file__).parent.joinpath("src")
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

# Import your backend functions (assumes these files are available in src/)
from data_cleaning import load_data, clean_data
from data_analysis import daily_returns, cumulative_returns, annualized_volatility, correlation_matrix, sharpe_ratio
from utils.plot_cumulative_returns import plot_cumulative_returns
from utils.plot_corr_matrix import plot_corr_matrix

# Configure the page
st.set_page_config(
    page_title="Stock Analytics",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="auto"
)

# Theme-aware/custom CSS (adapts for dark mode using prefers-color-scheme)
st.markdown(
    """
<style>
/* Basic spacing */
.main {
    padding: 1rem 1.5rem;
}

/* Buttons & inputs */
.stButton>button {
    width: 100%;
    padding: 0.45rem 0.9rem;
    border-radius: 6px;
    border: 1px solid transparent;
}

/* Insight box light theme */
.insight-box {
    background-color: #f0f2f6;
    color: #0b1437;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    border-left: 4px solid #ff4b4b;
    font-size: 0.95rem;
    line-height: 1.45;
}

/* Input fields styling */
.stTextInput>div>div>input, .stDateInput>div>div>input {
    background-color: #ffffff;
    color: inherit;
    border: 1px solid #d1d5db;
    padding: 0.35rem;
    border-radius: 6px;
}

/* Plotly containers should inherit text color from surrounding context */
.stPlotlyChart, .st-plotly-chart {
    color: inherit;
}

/* Dark mode adjustments */
@media (prefers-color-scheme: dark) {
    .insight-box {
        background-color: #111214;
        color: #e6eef8;
        border-left: 4px solid #ff6b6b;
    }
    .stTextInput>div>div>input, .stDateInput>div>div>input {
        background-color: #0b0b0c;
        color: #e6eef8;
        border: 1px solid #2f3336;
    }
    .stButton>button {
        background-color: #1f2224;
        color: #e6eef8;
        border: 1px solid #2f3336;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

def generate_insights(cum_ret, corr_matrix, ann_vol, sharpe, tickers):
    """Generate textual insights from computed metrics."""
    insights = []
    insights.append("## 📊 Cumulative Returns Analysis")

    # Defensive: ensure cum_ret has rows
    if cum_ret is None or cum_ret.empty:
        insights.append("No cumulative return data available for insight generation.")
        return "\n".join(insights)

    final_returns = cum_ret.iloc[-1]
    if not final_returns.empty:
        best_stock = final_returns.idxmax()
        worst_stock = final_returns.idxmin()
        best_return = final_returns.max() * 100
        worst_return = final_returns.min() * 100

        insights.append("**Performance Summary:**")
        insights.append(f"- 🏆 **{best_stock}** delivered the highest returns: **{best_return:.1f}%**")
        insights.append(f"- 📉 **{worst_stock}** showed the lowest returns: **{worst_return:.1f}%**")

        # Risk-adjusted performance (if sharpe provided)
        if sharpe is not None and not getattr(sharpe, "empty", True):
            best_sharpe = sharpe.idxmax()
            insights.append(f"- ⚖️ **{best_sharpe}** had the best risk-adjusted returns (Sharpe Ratio)")

    # Volatility
    insights.append("\n## ⚡ Volatility Analysis")
    if ann_vol is not None and not getattr(ann_vol, "empty", True):
        highest_vol = ann_vol.idxmax()
        lowest_vol = ann_vol.idxmin()
        vol_range = ann_vol.max() / ann_vol.min() if ann_vol.min() != 0 else np.inf

        insights.append("**Risk Assessment:**")
        insights.append(f"- 🎢 **{highest_vol}** is the most volatile (highest risk)")
        insights.append(f"- 🛌 **{lowest_vol}** is the most stable (lowest risk)")

        if vol_range > 2:
            insights.append("- ⚠️ Significant difference in risk profiles - consider diversification")

    # Correlation
    insights.append("\n## 🔗 Correlation & Diversification")
    if corr_matrix is not None and not corr_matrix.empty and corr_matrix.shape[0] > 1:
        strong_corr_pairs = []
        weak_corr_pairs = []

        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                stock_a = cols[i]
                stock_b = cols[j]
                corr_value = corr_matrix.iloc[i, j]
                if pd.isna(corr_value):
                    continue
                if corr_value > 0.7:
                    strong_corr_pairs.append((stock_a, stock_b, corr_value))
                elif corr_value < 0.3:
                    weak_corr_pairs.append((stock_a, stock_b, corr_value))

        if strong_corr_pairs:
            insights.append("**Strong Correlations (Move Together):**")
            for stock_a, stock_b, corr_val in strong_corr_pairs[:3]:
                insights.append(f"- 🔗 **{stock_a}** & **{stock_b}**: {corr_val:.2f} - High co-movement")

        if weak_corr_pairs:
            insights.append("**Diversification Opportunities:**")
            for stock_a, stock_b, corr_val in weak_corr_pairs[:3]:
                insights.append(f"- 🛡️ **{stock_a}** & **{stock_b}**: {corr_val:.2f} - Good for diversification")

        if not strong_corr_pairs and not weak_corr_pairs:
            insights.append("**Moderate correlations** - Balanced portfolio with some diversification benefits")

    # Sharpe ratio analysis
    insights.append("\n## 🎯 Risk-Adjusted Performance")
    if sharpe is not None and not getattr(sharpe, "empty", True):
        sharpe_pos = sharpe[sharpe > 0]
        sharpe_neg = sharpe[sharpe < 0]

        if len(sharpe_pos) > 0:
            best_sharpe_stock = sharpe_pos.idxmax()
            best_sharpe_value = sharpe_pos.max()
            insights.append("**Positive Risk-Adjusted Returns:**")
            insights.append(f"- ✅ {len(sharpe_pos)} stocks provided positive risk-adjusted returns")
            insights.append(f"- 🥇 **{best_sharpe_stock}** has the best Sharpe Ratio: **{best_sharpe_value:.2f}**")

        if len(sharpe_neg) > 0:
            insights.append("**Caution Required:**")
            insights.append(f"- ❌ {len(sharpe_neg)} stocks had negative risk-adjusted returns")

    # Overall portfolio assessment (avg correlation)
    insights.append("\n## 💼 Overall Portfolio Assessment")
    if corr_matrix is not None and not corr_matrix.empty and corr_matrix.shape[0] > 1:
        # compute average off-diagonal correlation
        tri_idxs = np.triu_indices_from(corr_matrix.values, k=1)
        if len(tri_idxs[0]) > 0:
            avg_correlation = np.nanmean(corr_matrix.values[tri_idxs])
            if avg_correlation < 0.4:
                insights.append("**🎉 Excellent Diversification** - Low average correlation provides strong risk reduction")
            elif avg_correlation < 0.7:
                insights.append("**👍 Good Diversification** - Moderate correlations offer reasonable risk management")
            else:
                insights.append("**⚠️ Limited Diversification** - High correlations mean stocks tend to move together")

    # Investment implications: match returns vs risk
    insights.append("\n## 💡 Investment Implications")
    if (final_returns is not None and not getattr(final_returns, "empty", True)
            and ann_vol is not None and not getattr(ann_vol, "empty", True)):
        high_return_high_risk = []
        high_return_low_risk = []

        for ticker in tickers:
            if ticker in final_returns.index and ticker in ann_vol.index:
                ret = final_returns[ticker]
                vol = ann_vol[ticker]
                if ret > final_returns.median() and vol < ann_vol.median():
                    high_return_low_risk.append(ticker)
                elif ret > final_returns.median() and vol > ann_vol.median():
                    high_return_high_risk.append(ticker)

        if high_return_low_risk:
            insights.append(f"**Quality Picks:** {', '.join(high_return_low_risk)} - High returns with below-average risk")
        if high_return_high_risk:
            insights.append(f"**High-Risk/High-Reward:** {', '.join(high_return_high_risk)} - Potential for high returns but with elevated risk")

    return "\n".join(insights)


def main():
    st.title("📈 Stock Performance Analytics")
    st.markdown("Analyze cumulative returns, correlations, and risk metrics for multiple stocks.")

    st.markdown("---")
    st.subheader("Input Parameters")

    # Multi-ticker input
    tickers_input = st.text_input(
        "Stock Tickers (comma-separated)",
        value="AAPL, TSLA, MSFT",
        placeholder="e.g., AAPL, MSFT, GOOGL, TSLA",
        help="Enter stock tickers separated by commas"
    )

    # Default date range: last 1 year -> today (so defaults update each day)
    today = datetime.date.today()
    default_start = today - datetime.timedelta(days=365)
    default_end = today

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=default_start)
    with col2:
        end_date = st.date_input("End Date", value=default_end)

    generate_clicked = st.button("Generate Analysis")

    if generate_clicked:
        if not tickers_input.strip():
            st.error("Please enter at least one stock ticker.")
            return

        # Validate date inputs
        if start_date >= end_date:
            st.error("Start date must be before end date.")
            return

        try:
            tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]
            # yfinance treats end date as exclusive. Add one day to include the user's end date.
            yf_end = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            yf_start = pd.to_datetime(start_date).strftime("%Y-%m-%d")

            with st.spinner("Loading and analyzing data..."):
                raw_data = load_data(tickers, yf_start, yf_end)
                cleaned_data = clean_data(raw_data)

                if cleaned_data is None or cleaned_data.empty:
                    st.error("No data found for the given tickers and date range.")
                    return

                # Find adjusted close columns robustly: common naming includes 'Adj Close', 'Close'
                close_candidates = [c for c in cleaned_data.columns if 'Adj Close' in c or 'AdjClose' in c or c.endswith('_Adj Close') or 'Close' in c]
                if not close_candidates:
                    st.error("No close/adjusted close columns found in the data.")
                    return

                adj_close = cleaned_data[close_candidates].copy()

                # Map columns to tickers: if the cleaned columns contain ticker prefix/suffix
                column_mapping = {}
                for col in adj_close.columns:
                    upper_col = col.upper()
                    matched = None
                    for ticker in tickers:
                        if ticker in upper_col:
                            matched = ticker
                            break
                    # fallback: keep original column name if no match
                    column_mapping[col] = matched if matched is not None else col

                adj_close.rename(columns=column_mapping, inplace=True)

                # If any duplicate ticker column names after mapping, keep first occurrence
                adj_close = adj_close.loc[:, ~adj_close.columns.duplicated()]

                st.success(f"✅ Successfully loaded data for {len(adj_close.columns)} stocks")

                # Calculate metrics
                daily_ret = daily_returns(adj_close)
                if daily_ret.empty:
                    st.error("Daily returns calculation produced no data.")
                    return

                cum_ret = cumulative_returns(adj_close)
                ann_vol = annualized_volatility(daily_ret)
                corr_matrix = correlation_matrix(daily_ret)
                sharpe = sharpe_ratio(daily_ret, ann_vol)

                # Visuals
                st.markdown("---")
                st.subheader("📈 Visual Analysis")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Cumulative Returns**")
                    fig_cumulative = plot_cumulative_returns(cum_ret)
                    st.plotly_chart(fig_cumulative, use_container_width=True)

                with col2:
                    st.markdown("**Correlation Matrix**")
                    fig_corr = plot_corr_matrix(corr_matrix)
                    st.plotly_chart(fig_corr, use_container_width=True)

                # Metrics table
                st.markdown("## 📋 Key Metrics")
                metrics_data = {
                    'Ticker': list(ann_vol.index),
                    'Annualized Volatility': [f"{v:.3f}" for v in ann_vol.values],
                    'Sharpe Ratio': [f"{v:.3f}" for v in sharpe.reindex(ann_vol.index).values]
                }
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True)

                # Insights
                st.markdown("## 🧠 Theoretical Analysis & Insights")
                insights = generate_insights(cum_ret, corr_matrix, ann_vol, sharpe, list(adj_close.columns))
                st.markdown(f'<div class="insight-box">{insights}</div>', unsafe_allow_html=True)

                # Interpretation helper
                with st.expander("📖 How to Interpret These Metrics"):
                    st.markdown(
                        """
- **Cumulative Returns**: Shows growth of $1 investment over time
- **Volatility**: Higher values = more risk/price swings
- **Sharpe Ratio**: >1 = Good, >2 = Excellent, <0 = Poor risk-adjusted returns
- **Correlation**: 
  - 0.8-1.0: Very strong (stocks move together)
  - 0.5-0.8: Strong correlation  
  - 0.3-0.5: Moderate correlation
  - 0.0-0.3: Weak correlation (good for diversification)
  - Negative: Stocks move in opposite directions
                        """
                    )

                # Prepare CSV for download
                output_data = cum_ret.copy()
                output_data = output_data.reset_index().rename(columns={'index': 'Date'}) if output_data.index.name is None else output_data.reset_index()
                csv = output_data.to_csv(index=False)

                st.download_button(
                    label="📥 Download CSV Report",
                    data=csv,
                    file_name="stock_analysis_report.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()