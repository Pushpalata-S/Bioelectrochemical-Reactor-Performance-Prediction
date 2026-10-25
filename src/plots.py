"""Interactive Plotly chart generation with a consistent scientific palette and styling."""
from typing import Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import COLORS


def apply_sci_layout(fig: go.Figure, title: str, xaxis_title: str, yaxis_title: str, height: int = 420) -> go.Figure:
    """Apply consistent scientific theme layout to any Plotly figure."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=15, color=COLORS["navy_dark"], family="Inter, sans-serif"),
            x=0.01,
            y=0.96,
        ),
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(size=12, color=COLORS["text_dark"])),
            gridcolor=COLORS["slate_border"],
            zerolinecolor=COLORS["slate_border"],
            linecolor=COLORS["slate_border"],
            tickfont=dict(size=11, color=COLORS["slate_text"]),
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(size=12, color=COLORS["text_dark"])),
            gridcolor=COLORS["slate_border"],
            zerolinecolor=COLORS["slate_border"],
            linecolor=COLORS["slate_border"],
            tickfont=dict(size=11, color=COLORS["slate_text"]),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=50, r=30, t=50, b=50),
        height=height,
        font=dict(family="Inter, sans-serif"),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=COLORS["slate_border"],
            borderwidth=1,
            font=dict(size=11),
        ),
    )
    return fig


def plot_missingness(df_missing: pd.DataFrame) -> go.Figure:
    """Plot missingness percentage across features and physics parameters."""
    df_sorted = df_missing.sort_values("Missing_Pct", ascending=True).copy()
    
    colors = [
        COLORS["red"] if pct > 80 else (COLORS["amber"] if pct > 40 else COLORS["teal_primary"])
        for pct in df_sorted["Missing_Pct"]
    ]

    fig = go.Figure(
        go.Bar(
            x=df_sorted["Missing_Pct"],
            y=df_sorted["Feature"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="#FFFFFF", width=1)),
            text=[f"{p:.1f}% ({m}/{m+pr})" for p, m, pr in zip(df_sorted["Missing_Pct"], df_sorted["Missing"], df_sorted["Present"])],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Missing: %{x:.1f}%<extra></extra>",
        )
    )
    fig = apply_sci_layout(fig, "Feature Missingness & Excluded Physics Profile", "Missingness (%)", "Feature", height=420)
    fig.update_xaxes(range=[0, 115])
    return fig


def plot_parity(
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    title: str,
    x_label: str,
    y_label: str,
    is_log: bool = False,
    task: str = "electricity",
) -> go.Figure:
    """
    Parity plot (Actual vs. Predicted) with 1:1 reference line.
    Supports log10 scaling for power density or linear for COD removal %.
    """
    plot_df = df.dropna(subset=[actual_col, pred_col]).copy()
    
    fig = go.Figure()

    # Scatter points
    fig.add_trace(
        go.Scatter(
            x=plot_df[actual_col],
            y=plot_df[pred_col],
            mode="markers",
            marker=dict(
                size=9,
                color=COLORS["teal_primary"],
                opacity=0.85,
                line=dict(width=1, color=COLORS["navy_dark"]),
            ),
            text=plot_df.get("Study_ID", plot_df.get("Row_ID", "")),
            customdata=plot_df["Row_ID"],
            hovertemplate="<b>%{customdata}</b> (%{text})<br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>",
            name="Reactor Observations",
        )
    )

    # 1:1 diagonal line
    if is_log:
        min_val = max(1e-2, min(plot_df[actual_col].min(), plot_df[pred_col].min())) * 0.7
        max_val = max(plot_df[actual_col].max(), plot_df[pred_col].max()) * 1.4
        line_vals = [min_val, max_val]
        
        fig.add_trace(
            go.Scatter(
                x=line_vals,
                y=line_vals,
                mode="lines",
                line=dict(color=COLORS["navy_dark"], dash="dash", width=1.5),
                name="1:1 Parity Line",
            )
        )
        # Factor of 3 error envelope lines for electricity
        fig.add_trace(
            go.Scatter(
                x=line_vals,
                y=[v * 3.0 for v in line_vals],
                mode="lines",
                line=dict(color=COLORS["slate_text"], dash="dot", width=1),
                name="+3x Factor Bound",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=line_vals,
                y=[v / 3.0 for v in line_vals],
                mode="lines",
                line=dict(color=COLORS["slate_text"], dash="dot", width=1),
                name="-3x Factor Bound",
            )
        )
        fig.update_xaxes(type="log", range=[np.log10(min_val), np.log10(max_val)])
        fig.update_yaxes(type="log", range=[np.log10(min_val), np.log10(max_val)])
    else:
        line_vals = [0, 100]
        fig.add_trace(
            go.Scatter(
                x=line_vals,
                y=line_vals,
                mode="lines",
                line=dict(color=COLORS["navy_dark"], dash="dash", width=1.5),
                name="1:1 Parity Line",
            )
        )
        # +/- 10% pts envelope lines for COD removal
        fig.add_trace(
            go.Scatter(
                x=line_vals,
                y=[v + 10 for v in line_vals],
                mode="lines",
                line=dict(color=COLORS["slate_text"], dash="dot", width=1),
                name="+10% pts Bound",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=line_vals,
                y=[v - 10 for v in line_vals],
                mode="lines",
                line=dict(color=COLORS["slate_text"], dash="dot", width=1),
                name="-10% pts Bound",
            )
        )
        fig.update_xaxes(range=[0, 105])
        fig.update_yaxes(range=[0, 105])

    fig = apply_sci_layout(fig, title, x_label, y_label, height=450)
    return fig


def plot_residuals(
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    title: str,
    x_label: str,
    y_label: str,
    is_log: bool = False,
) -> go.Figure:
    """Plot residuals (Actual - Predicted) vs Actual."""
    plot_df = df.dropna(subset=[actual_col, pred_col]).copy()
    
    if is_log:
        y_act = np.log10(plot_df[actual_col].astype(float))
        y_pred = np.log10(plot_df[pred_col].astype(float))
        res = y_act - y_pred
        x_vals = y_act
    else:
        res = plot_df[actual_col] - plot_df[pred_col]
        x_vals = plot_df[actual_col]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=res,
            mode="markers",
            marker=dict(
                size=8,
                color=COLORS["navy_dark"],
                opacity=0.8,
                line=dict(width=1, color=COLORS["teal_primary"]),
            ),
            text=plot_df.get("Row_ID", ""),
            hovertemplate="<b>%{text}</b><br>Actual: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>",
            name="Residuals",
        )
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["amber"], line_width=1.5)

    fig = apply_sci_layout(fig, title, x_label, y_label, height=380)
    return fig


def plot_benchmark_r2(benchmark_df: pd.DataFrame, task_name: str) -> go.Figure:
    """Horizontal bar chart of R2 (Mean +/- SD) across 11 benchmarked algorithms."""
    df_sorted = benchmark_df.sort_values("R2_mean", ascending=True).copy()
    
    colors = [
        COLORS["teal_primary"] if "Random Forest" in m or "Gaussian Process" in m
        else (COLORS["amber"] if "Baseline" in m else COLORS["navy_card"])
        for m in df_sorted["Model"]
    ]

    fig = go.Figure(
        go.Bar(
            x=df_sorted["R2_mean"],
            y=df_sorted["Model"],
            orientation="h",
            error_x=dict(
                type="data",
                array=df_sorted["R2_sd"],
                visible=True,
                color=COLORS["slate_text"],
                thickness=1.2,
                width=4,
            ),
            marker=dict(color=colors, line=dict(color="#FFFFFF", width=0.8)),
            text=[f"{r:.3f}" for r in df_sorted["R2_mean"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>R²: %{x:.3f} ± SD<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_dash="solid", line_color="#000000", line_width=1)

    fig = apply_sci_layout(
        fig,
        f"{task_name}: Study-Grouped Cross-Validation R² Comparison",
        "R² (Mean ± SD on Unseen Studies)",
        "Estimator",
        height=450,
    )
    return fig


def plot_benchmark_rmse(benchmark_df: pd.DataFrame, task_name: str, unit: str) -> go.Figure:
    """Horizontal bar chart of RMSE across benchmarked algorithms."""
    df_sorted = benchmark_df.sort_values("RMSE_mean", ascending=False).copy()
    
    colors = [
        COLORS["teal_primary"] if r == 1 else (COLORS["amber"] if "Baseline" in m else COLORS["navy_card"])
        for r, m in zip(df_sorted["Rank"], df_sorted["Model"])
    ]

    fig = go.Figure(
        go.Bar(
            x=df_sorted["RMSE_mean"],
            y=df_sorted["Model"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="#FFFFFF", width=0.8)),
            text=[f"{r:.2f} {unit}" for r in df_sorted["RMSE_mean"]],
            textposition="inside",
            hovertemplate="<b>%{y}</b><br>RMSE: %{x:.3f} " + unit + "<extra></extra>",
        )
    )

    fig = apply_sci_layout(
        fig,
        f"{task_name}: Root Mean Squared Error (RMSE)",
        f"RMSE ({unit})",
        "Estimator",
        height=450,
    )
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, task_name: str) -> go.Figure:
    """Horizontal bar chart of permutation feature importance."""
    df_sorted = importance_df.sort_values("Importance", ascending=True).copy()
    unit = df_sorted["Unit"].iloc[0]

    fig = go.Figure(
        go.Bar(
            x=df_sorted["Importance"],
            y=df_sorted["Feature"],
            orientation="h",
            error_x=dict(
                type="data",
                array=df_sorted["SD"],
                visible=True,
                color=COLORS["slate_text"],
                thickness=1.2,
                width=4,
            ),
            marker=dict(color=COLORS["teal_primary"], line=dict(color=COLORS["navy_dark"], width=1)),
            text=[f"+{imp:.3f} {unit}" for imp in df_sorted["Importance"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>RMSE Increase: +%{x:.3f} ± SD<extra></extra>",
        )
    )

    fig = apply_sci_layout(
        fig,
        f"{task_name}: Permutation Feature Importance",
        f"Mean Increase in Test RMSE ({unit})",
        "Design Feature",
        height=380,
    )
    return fig


def plot_data_leakage(leakage_df: pd.DataFrame) -> go.Figure:
    """Visual proof of data leakage: Study-Grouped CV vs. Random CV."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Study-Grouped CV (Honest Generalization)",
            x=[f"{row['Task']}: {row['Model']}" for _, row in leakage_df.iterrows()],
            y=leakage_df["Grouped_R2"],
            marker_color=COLORS["teal_primary"],
            text=[f"{v:.3f}" for v in leakage_df["Grouped_R2"]],
            textposition="outside",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Random CV (Same-Paper Leakage)",
            x=[f"{row['Task']}: {row['Model']}" for _, row in leakage_df.iterrows()],
            y=leakage_df["Random_R2"],
            marker_color=COLORS["amber"],
            text=[f"{v:.3f}" for v in leakage_df["Random_R2"]],
            textposition="outside",
        )
    )

    fig.add_hline(y=0, line_dash="solid", line_color="#000000", line_width=1)
    fig.update_layout(barmode="group")

    fig = apply_sci_layout(
        fig,
        "The Data Leakage Effect: Grouped vs. Random Cross-Validation",
        "Benchmarked Task & Algorithm",
        "Cross-Validated R²",
        height=450,
    )
    return fig


def plot_target_distribution(df: pd.DataFrame, col: str, title: str, x_label: str, is_log: bool = False) -> go.Figure:
    """Distribution histogram of reactor target variables."""
    valid_data = df[col].dropna()
    if is_log:
        valid_data = valid_data[valid_data > 0]

    fig = px.histogram(
        valid_data,
        x=col,
        nbins=25,
        color_discrete_sequence=[COLORS["teal_primary"]],
        marginal="box",
        log_x=is_log,
    )

    fig = apply_sci_layout(fig, title, x_label, "Reactor Count", height=380)
    return fig


def plot_bivariate(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    title: str,
    x_label: str,
    y_label: str,
    log_x: bool = False,
    log_y: bool = False,
) -> go.Figure:
    """Bivariate scatter plot with scientific color encoding."""
    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    if log_x:
        plot_df = plot_df[plot_df[x_col] > 0]
    if log_y:
        plot_df = plot_df[plot_df[y_col] > 0]

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color=color_col,
        hover_data=["Row_ID", "Study_ID"],
        log_x=log_x,
        log_y=log_y,
        color_discrete_sequence=px.colors.qualitative.Dark24,
    )

    fig = apply_sci_layout(fig, title, x_label, y_label, height=450)
    return fig


def plot_scenario_tradeoff(scenarios_df: pd.DataFrame) -> go.Figure:
    """Scatter trade-off plot: Power Density vs COD Removal for candidate design scenarios."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=scenarios_df["Pred_COD_Removal_pct"],
            y=scenarios_df["Pred_Power_Density_mW_m2"],
            mode="markers+text",
            text=scenarios_df["Scenario_Name"],
            textposition="top center",
            marker=dict(
                size=14,
                color=scenarios_df["Pred_Power_Density_mW_m2"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Power (mW/m²)"),
                line=dict(color=COLORS["navy_dark"], width=1.5),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Predicted COD Removal: %{x:.1f}%<br>"
                "Predicted Power Density: %{y:.2f} mW/m²<extra></extra>"
            ),
        )
    )

    fig = apply_sci_layout(
        fig,
        "Design Scenarios: Simultaneous Power Generation vs. COD Removal",
        "Predicted COD Removal (%)",
        "Predicted Power Density (mW/m²)",
        height=450,
    )
    return fig
