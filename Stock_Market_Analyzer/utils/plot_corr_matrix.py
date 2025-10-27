import plotly.graph_objects as go
import plotly.express as px

def plot_corr_matrix(corr_matrix):
    """
    Create a Plotly heatmap for correlation matrix
    """
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        hoverongaps=False,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 12}
    ))
    
    fig.update_layout(
        title="Correlation Matrix of Stock Returns",
        xaxis_title="Stocks",
        yaxis_title="Stocks",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='black'),
        height=400
    )
    
    return fig