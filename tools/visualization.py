"""This module contains functions for visualization"""

import warnings

import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")


def logview(
    df_log: pd.DataFrame,
    features_to_log: list = [],
    col_depth: str = "DEPTH",
) -> plotly.subplots.make_subplots:
    """
    function to construct layout for the well
    Args:
        df_log ():
        df_lith_mixed ():
        df_lith_dominant ():
        df_formation ():
        features_to_log ():
        col_depth ():

    Returns:

    """
    # Base number of columns is the number of features in df_log
    num_cols = len(features_to_log) if features_to_log else df_log.shape[1]

    fig = make_subplots(rows=1, cols=num_cols, shared_yaxes=True)

    # specify features to plot
    features = [col for col in df_log.columns if col not in [col_depth] + ["WELL"]]

    # cur subplot pos
    col_numbers = 0

    # plotting features
    for ix, feat in enumerate(features):
        fig.add_trace(
            go.Scatter(
                x=df_log[feat],
                y=df_log[col_depth],
                mode="lines",
                line=dict(color="black", width=0.5),
                name=feat,
            ),
            row=1,
            col=ix + 1,
        )

        fig.update_xaxes(
            title=dict(text=feat),
            row=1,
            col=ix + 1,
            side="top",
            tickangle=-90,
        )
        if feat in features_to_log:
            fig.update_xaxes(col=ix + 1, type="log")

        col_numbers += 1

    fig.update_yaxes(
        title_text="DEPTH",
        row=1,
        col=1,
        autorange="reversed",
        tickformat=".0f",
    )
    fig.update_layout(height=900, width=1200, showlegend=False)

    return fig
