import pandas as pd
import streamlit as st

from corpus_metrics import compute_frequency_dict, compute_general_metrics, compute_vector_map, compute_vocabulary_growth
from data_loader import load_cluster_map
from visualizations import build_mayak_semantic_map, render_vector_map_chart


def render_corpus_tab(full_corpus, min_year, max_year):
    st.markdown("## Статистика корпуса")

    stats_year_range = st.slider(
        "Период",
        min_year,
        max_year,
        (min_year, max_year),
        key="stats_year_range",
    )

    filtered_corpus_stats = [
        item for item in full_corpus
        if stats_year_range[0] <= item["year_finished"] <= stats_year_range[1]
    ]

    if not filtered_corpus_stats:
        st.warning("Нет данных за выбранный период.")
        return

    tab_corp_metrics, tab_corp_freq, tab_corp_growth = st.tabs([
        "Общие метрики",
        "Частотный словарь",
        "Рост словаря",
    ])

    with tab_corp_metrics:
        metrics = compute_general_metrics(filtered_corpus_stats)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Произведений", metrics["total_texts"])
        col2.metric("Предложений", f"{metrics['total_tokens']:,}".replace(",", "\u00a0"))
        col3.metric("Лемм всего", f"{metrics['total_lemmas']:,}".replace(",", "\u00a0"))
        col4.metric("Уникальных лемм", f"{metrics['unique_lemmas']:,}".replace(",", "\u00a0"))

        st.divider()
        st.subheader("Тексты по годам")
        year_texts_df = pd.DataFrame(
            sorted(metrics["texts_by_year"].items()),
            columns=["Год", "Текстов"],
        ).set_index("Год")
        st.bar_chart(year_texts_df)

    with tab_corp_freq:
        col_sw, col_n = st.columns([1, 2])
        with col_sw:
            exclude_sw = st.checkbox("Исключить стоп-слова", value=True, key="freq_exclude_sw")
        with col_n:
            top_n = st.slider("Топ-N слов", 10, 300, 50, key="freq_top_n")

        freq_counter = compute_frequency_dict(filtered_corpus_stats, exclude_stopwords=exclude_sw)
        total_lemmas_count = sum(freq_counter.values())
        top_lemmas = freq_counter.most_common(top_n)

        freq_df = pd.DataFrame(top_lemmas, columns=["Лемма", "Частота"])
        freq_df["% от корпуса"] = (freq_df["Частота"] / total_lemmas_count * 100).round(3)
        freq_df.index = range(1, len(freq_df) + 1)

        col_table, col_chart = st.columns([1, 1.2])
        with col_table:
            st.dataframe(freq_df, width="stretch")
        with col_chart:
            st.bar_chart(freq_df.set_index("Лемма")["Частота"].head(30))

    with tab_corp_growth:
        growth_data = compute_vocabulary_growth(filtered_corpus_stats)
        growth_df = pd.DataFrame(growth_data).set_index("Год")

        if not growth_df.empty:
            st.subheader("Накопленный словарный запас")
            st.caption("Сколько уникальных лемм встречено в корпусе к каждому году")
            st.line_chart(growth_df[["Уникальных лемм накоплено"]])

            st.divider()
            st.subheader("Новых уникальных лемм в год")
            st.caption("Сколько ранее не встречавшихся лемм появилось в текстах каждого года")
            st.bar_chart(growth_df[["Новых лемм"]])

            st.divider()
            st.subheader("Type-Token Ratio")
            st.caption("Отношение уникальных лемм к общему числу лемм в году")
            st.line_chart(growth_df[["Type-Token Ratio"]])

    st.divider()
    st.title("Векторная карта самых частотных слов")
    st.info("Основана на векторных представлениях слов из модели Navec, обученной на корпусе русской литературы.")

    col_vm_sw, col_vm_n = st.columns([1, 2])
    with col_vm_sw:
        vm_exclude_sw = st.checkbox("Исключить стоп-слова", value=True, key="vm_exclude_sw")
    with col_vm_n:
        vm_top_n = st.slider("Количество слов", 20, 300, 100, key="vm_top_n")

    map_df = compute_vector_map(
        filtered_corpus_stats,
        top_n=vm_top_n,
        exclude_stopwords=vm_exclude_sw,
        pca_base_size=max(300, vm_top_n),
    )

    if map_df is None:
        st.warning("Недостаточно слов с векторными представлениями для построения карты.")
    else:
        render_vector_map_chart(map_df)

    st.divider()
    st.title("Семантическая карта по Индексу Маяка")

    cluster_map_df = load_cluster_map()
    if cluster_map_df is None:
        st.info(
            "📊 Карта контекстуальных кластеров ещё не сгенерирована. "
            "Её нужно предрассчитать из корня проекта:\n\n"
            "```bash\npython -m src.map_builder\n```\n\n"
            "Это займёт 20–30 минут в первый раз (300 слов × 20 кластеров)."
        )
        return

    col_search, _, col_size = st.columns([0.8, 0.02, 0.18], gap="small")
    with col_search:
        sm_search_word = st.text_input(
            "🔍 Найти слово:",
            placeholder="Например: революция, любовь, рабочий...",
            key="semantic_map_search",
        )
    with col_size:
        sm_size_by_freq = st.checkbox(
            "Размер ~ частота",
            value=True,
            key="semantic_map_size_freq",
            help="Крупнее = чаще встречается",
        )

    search_result = None
    if sm_search_word:
        match = cluster_map_df[cluster_map_df["word"] == sm_search_word.strip().lower()]
        if not match.empty:
            search_result = match.iloc[0]

    if sm_search_word:
        if search_result is None:
            st.info(f"Слово «{sm_search_word}» не входит в топ-300 слов карты.")
        else:
            st.success(
                f"**{search_result['word']}** — Кластер {int(search_result['cluster'])}, "
                f"частота: {int(search_result['freq'])}, "
                f"координаты: ({search_result['x']:.2f}, {search_result['y']:.2f})"
            )

    selected_clusters = sorted(cluster_map_df["cluster"].unique().tolist())
    fig = build_mayak_semantic_map(cluster_map_df, selected_clusters, sm_size_by_freq)
    st.plotly_chart(fig, width="stretch")

    with st.expander("Состав кластеров"):
        display_df = (
            cluster_map_df[cluster_map_df["cluster"].isin(selected_clusters)][["cluster", "word", "freq"]]
            .rename(columns={"cluster": "Кластер", "word": "Слово", "freq": "Частота"})
            .sort_values(["Кластер", "Частота"], ascending=[True, False])
            .reset_index(drop=True)
        )
        display_df.index = display_df.index + 1
        st.dataframe(display_df, width="stretch")
