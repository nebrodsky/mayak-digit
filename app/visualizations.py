import altair as alt
import plotly.graph_objects as go
import streamlit as st


def build_mayak_semantic_map(df, selected_clusters, size_by_freq):
    fig = go.Figure()

    cluster_palette = [
        "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
        "#9B5DE5", "#00BBF9", "#F15BB5", "#06D6A0", "#FB5607",
        "#8338EC", "#3A86FF", "#FFBE0B", "#FF006E", "#8AC926",
        "#DC2F02", "#370617", "#03071E", "#D62828", "#F77F00",
    ]

    all_clusters = sorted(df["cluster"].unique())

    for cluster_id in all_clusters:
        if cluster_id not in selected_clusters:
            continue

        subset = df[df["cluster"] == cluster_id]
        color = cluster_palette[(cluster_id - 1) % len(cluster_palette)]

        if size_by_freq:
            freq_vals = subset["freq"].values.astype(float)
            max_freq = freq_vals.max() if freq_vals.max() > 0 else 1
            sizes = 8 + (freq_vals / max_freq) * 20
        else:
            sizes = 14

        fig.add_trace(
            go.Scatter(
                x=subset["x"],
                y=subset["y"],
                mode="markers+text",
                name=f"Кластер {cluster_id}",
                text=subset["word"],
                textposition="top center",
                textfont=dict(size=11, family="Arial"),
                marker=dict(
                    size=sizes,
                    color=color,
                    opacity=0.82,
                    line=dict(width=0.8, color="white"),
                ),
                customdata=subset[["word", "cluster", "freq"]].values,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Кластер: %{customdata[1]}<br>"
                    "Частота: %{customdata[2]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text="Семантическая карта корпуса Маяковского<br><sup>Индекс Маяка · UMAP · K-Means</sup>",
            font=dict(size=16, color="#e0e0e0"),
            x=0.5,
        ),
        legend=dict(
            title="Кластеры",
            itemsizing="constant",
            bgcolor="rgba(45, 45, 45, 0.85)",
            bordercolor="#555555",
            borderwidth=1,
            font=dict(color="#e0e0e0"),
            title_font=dict(color="#e0e0e0"),
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#0d0d0d",
        margin=dict(l=20, r=20, t=80, b=20),
        height=600,
        hovermode="closest",
    )

    return fig


def render_vector_map_chart(map_df):
    points = alt.Chart(map_df).mark_circle(opacity=0.7).encode(
        x=alt.X("x:Q", axis=alt.Axis(labels=False, ticks=False, title=None, grid=False)),
        y=alt.Y("y:Q", axis=alt.Axis(labels=False, ticks=False, title=None, grid=False)),
        size=alt.Size("Частота:Q", scale=alt.Scale(range=[40, 400]), legend=None),
        color=alt.Color("Частота:Q", scale=alt.Scale(scheme="viridis"), legend=None),
        tooltip=["Слово:N", "Частота:Q"],
    )
    labels = alt.Chart(map_df).mark_text(dx=6, dy=-6, fontSize=11, align="left", color="white").encode(
        x="x:Q",
        y="y:Q",
        text="Слово:N",
        tooltip=["Слово:N", "Частота:Q"],
    )
    st.altair_chart(
        (points + labels).properties(height=600).configure_view(strokeWidth=0),
        width="stretch",
    )
