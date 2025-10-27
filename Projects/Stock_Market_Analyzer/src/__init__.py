import streamlit as st
import pandas as pd
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Import your existing functions
from data_cleaning import load_data, clean_data
from src.data_analysis import daily_returns, cumulative_returns, annualized_volatility, correlation_matrix, sharpe_ratio

# Import plotting functions (you'll need to create these in Plotly)
from utils.plot_cumulative_returns import plot_cumulative_returns
from utils.plot_corr_matrix import plot_correlation_heatmap

# Configure page
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for minimalist styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        width: 100%;
        background-color: #2E2E2E;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
    }
    .stButton button:hover {
        background-color: #404040;
    }
    .stTextInput input {
        background-color: #1E1E1E;
        color: white;
        border: 1px solid #404040;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.title("📈 Stock Performance Analyzer")
    st.markdown("---")
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        tickers_input = st.text_input(
            "Stock Tickers",
            placeholder="Enter tickers separated by commas (e.g., AAPL, MSFT, GOOGL)",
            help="Enter stock symbols separated by commas"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_btn = st.button("Generate Analysis", type="primary")
    
    # Date range
    col3, col4 = st.columns(2)
    with col3:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
    with col4:
        end_date = st.date_input("End Date", value=pd.to_datetime("2025-10-16"))
    
    # Process when generate is clicked
    if generate_btn and tickers_input:
        try:
            with st.spinner("Generating analysis..."):
                # Parse tickers
                tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]
                
                # Load and clean data
                raw_data = load_data(tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                cleaned_data = clean_data(raw_data)
                
                # Extract adjusted close prices
                adj_close = cleaned_data.filter(like='Close')
                
                # Calculate metrics
                daily_ret = daily_returns(adj_close)
                cum_ret = cumulative_returns(adj_close)
                ann_vol = annualized_volatility(daily_ret)
                corr_matrix = correlation_matrix(daily_ret)
                sharpe = sharpe_ratio(daily_ret, ann_vol)
                
                # Create metrics dataframe for CSV export
                metrics_df = pd.DataFrame({
                    'Annualized_Volatility': ann_vol,
                    'Sharpe_Ratio': sharpe
                })
                
                # Display results section
                st.markdown("---")
                st.header("Analysis Results")
                
                # Charts in columns
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.subheader("Cumulative Returns")
                    fig_cum = plot_cumulative_returns(cum_ret)
                    st.plotly_chart(fig_cum, use_container_width=True)
                
                with col_chart2:
                    st.subheader("Correlation Matrix")
                    fig_corr = plot_correlation_heatmap(corr_matrix)
                    st.plotly_chart(fig_corr, use_container_width=True)
                
                # Metrics and download section
                st.markdown("---")
                col_metrics, col_download = st.columns([2, 1])
                
                with col_metrics:
                    st.subheader("Performance Metrics")
                    
                    # Display metrics in a clean format
                    metrics_col1, metrics_col2 = st.columns(2)
                    
                    with metrics_col1:
                        st.write("**Annualized Volatility**")
                        for ticker, vol in ann_vol.items():
                            st.write(f"{ticker}: {vol:.4f}")
                    
                    with metrics_col2:
                        st.write("**Sharpe Ratio**")
                        for ticker, ratio in sharpe.items():
                            st.write(f"{ticker}: {ratio:.4f}")
                
                with col_download:
                    st.subheader("Export Data")
                    
                    # Prepare CSV for download
                    csv_data = metrics_df.to_csv(index=True)
                    
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name="stock_analysis_metrics.csv",
                        mime="text/csv",
                        help="Download calculated metrics as CSV"
                    )
                
                # Store data in session state for potential reuse
                st.session_state['analysis_data'] = {
                    'cumulative_returns': cum_ret,
                    'correlation_matrix': corr_matrix,
                    'metrics': metrics_df
                }
                
        except Exception as e:
            st.error(f"Error generating analysis: {str(e)}")
            st.info("Please check your ticker symbols and try again.")
    
    elif generate_btn and not tickers_input:
        st.warning("Please enter at least one stock ticker.")

if __name__ == "__main__":
    main()
