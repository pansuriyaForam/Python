import plotly.graph_objects as go
import plotly.express as px

def plot_cumulative_returns(cum_returns):
    """
    Create a Plotly line chart for cumulative returns
    """
    fig = go.Figure()
    
    for column in cum_returns.columns:
        fig.add_trace(go.Scatter(
            x=cum_returns.index,
            y=cum_returns[column],
            name=column,
            mode='lines'
        ))
    
    fig.update_layout(
        title="Cumulative Returns Over Time",
        xaxis_title="Date",
        yaxis_title="Cumulative Returns",
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='black'),
        height=400
    )
    
    fig.update_xaxes(gridcolor='lightgray')
    fig.update_yaxes(gridcolor='lightgray')
    
    return fig