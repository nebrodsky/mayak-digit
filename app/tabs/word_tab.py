import pandas as pd
import streamlit as st

from llm_integration import render_llm_section
from src.analyzer import calculate_delta_analysis
from ui_helpers import display_contexts_table_highlighted, display_contexts_table_simple
from word_analysis import build_word_analysis_state


def _render_synonyms(compare_periods, synonyms, synonyms_filtered):
    st.subheader("Семантический кластер")

    if not synonyms:
        st.write("Семантически близкие слова не найдены или слово отсутствует в модели.")
        return

    if compare_periods:
        st.info("Подсчет семантически близких слов (векторных синонимов) происходит без привязки к периоду (на основе общего векторного словаря и полного корпуса Маяковского)")

    show_coefficients = st.checkbox("Показать коэффициенты близости", value=True)
    if show_coefficients:
        synonyms_str = ", ".join([f"{syn} ({score:.4f})" for syn, score in synonyms])
        st.write(f"Семантически близкие слова по общему корпусу художественной литературы (с коэффициентами близости): {synonyms_str}")
    else:
        st.write(f"Семантически близкие слова по общему корпусу художественной литературы (без коэффициентов): {', '.join([syn for syn, _ in synonyms])}")

    st.info("Список включает слова из общего векторного словаря, которые могут не встречаться в поэтических текстах Маяковского. Ниже — только те слова, которые действительно есть в корпусе")
    st.write(f"Семантически близкие слова, найденные в корпусе: {', '.join(synonyms_filtered)}")


def _render_period_metrics(compare_periods, year_range, year_range_2, results, results_2, count_stopwords):
    total_occurrences = results["total_occurrences"]
    year_dist = results["year_dist"]
    pos_dist = results["pos_dist"]

    if compare_periods and results_2:
        total_occurrences_2 = results_2["total_occurrences"]
        year_dist_2 = results_2["year_dist"]
        pos_dist_2 = results_2["pos_dist"]

        st.subheader(f"📍 Период {year_range[0]} — {year_range[1]}")
        col1_metric, col1_pos, col1_years = st.columns([1, 1.2, 1.2])
        with col1_metric:
            st.metric("Всего употреблений", total_occurrences)
        with col1_pos:
            st.caption("Частеречное окружение")
            pos_data = pos_dist["with_stopwords"] if count_stopwords else pos_dist["filtered"]
            pos_df = pd.DataFrame(pos_data.items(), columns=["Часть речи", "Кол-во"])
            st.bar_chart(pos_df.set_index("Часть речи"), height=200)
        with col1_years:
            st.caption("Динамика")
            year_df = pd.DataFrame(year_dist.items(), columns=["Год", "Частота"]).sort_values("Год")
            st.line_chart(year_df.set_index("Год"), height=200)

        st.subheader(f"📍 Период {year_range_2[0]} — {year_range_2[1]}")
        col2_metric, col2_pos, col2_years = st.columns([1, 1.2, 1.2])
        with col2_metric:
            st.metric("Всего употреблений", total_occurrences_2)
        with col2_pos:
            st.caption("Частеречное окружение")
            pos_data_2 = pos_dist_2["with_stopwords"] if count_stopwords else pos_dist_2["filtered"]
            pos_df_2 = pd.DataFrame(pos_data_2.items(), columns=["Часть речи", "Кол-во"])
            st.bar_chart(pos_df_2.set_index("Часть речи"), height=200)
        with col2_years:
            st.caption("Динамика")
            year_df_2 = pd.DataFrame(year_dist_2.items(), columns=["Год", "Частота"]).sort_values("Год")
            st.line_chart(year_df_2.set_index("Год"), height=200)

        st.divider()
        occ_delta = total_occurrences_2 - total_occurrences
        occ_pct = occ_delta / max(total_occurrences, 1) * 100
        col_delta, _, _ = st.columns(3)
        with col_delta:
            st.metric("Δ Употреблений", f"{occ_delta:+d}", f"{occ_pct:+.1f}%")
        return

    col_metric, col_pos, col_years = st.columns(3)
    with col_metric:
        st.subheader("Статистика")
        st.metric("Всего употреблений", total_occurrences)
    with col_pos:
        st.subheader("Частеречное окружение")
        pos_data = pos_dist["with_stopwords"] if count_stopwords else pos_dist["filtered"]
        pos_df = pd.DataFrame(pos_data.items(), columns=["Часть речи", "Кол-во"])
        st.bar_chart(pos_df.set_index("Часть речи"))
    with col_years:
        st.subheader("Динамика")
        year_df = pd.DataFrame(year_dist.items(), columns=["Год", "Частота"]).sort_values("Год")
        st.line_chart(year_df.set_index("Год"))


def _render_index_progress(weights_df):
    if weights_df.empty:
        return

    max_val = weights_df["Индекс"].max()
    weights_df["Сила связи"] = weights_df["Индекс"] / max_val
    weights_df.index = range(1, len(weights_df) + 1)
    st.dataframe(
        weights_df[["Лемма", "Сила связи"]],
        column_config={
            "Сила связи": st.column_config.ProgressColumn(
                "Контекстуальная близость",
                format="%.2f",
                min_value=0,
                max_value=1,
            )
        },
        width="stretch",
    )


def _render_semantic_connections(compare_periods, year_range, year_range_2, results, results_2, count_stopwords):
    top_neighbors = results["window_neighbors"]
    proximity_weights = results["proximity_weights"]

    st.subheader("Семантические связи")
    if compare_periods and results_2:
        top_neighbors_2 = results_2["window_neighbors"]
        proximity_weights_2 = results_2["proximity_weights"]
        delta_analysis = calculate_delta_analysis(results, results_2, count_stopwords=count_stopwords)

        tab_window, tab_index, tab_delta = st.tabs(["Классическое окно", "Индекс Маяка", "Дельта-анализ"])

        with tab_window:
            col_wnd_1, col_wnd_2 = st.columns(2)
            with col_wnd_1:
                st.caption(f"Период {year_range[0]}—{year_range[1]}")
                data = top_neighbors["with_stopwords"] if count_stopwords else top_neighbors["filtered"]
                n_df = pd.DataFrame(data.most_common(10), columns=["Лемма", "Частота"])
                n_df.index = range(1, len(n_df) + 1)
                st.table(n_df)
            with col_wnd_2:
                st.caption(f"Период {year_range_2[0]}—{year_range_2[1]}")
                data_2 = top_neighbors_2["with_stopwords"] if count_stopwords else top_neighbors_2["filtered"]
                n_df_2 = pd.DataFrame(data_2.most_common(10), columns=["Лемма", "Частота"])
                n_df_2.index = range(1, len(n_df_2) + 1)
                st.table(n_df_2)

        with tab_index:
            col_idx_1, col_idx_2 = st.columns(2)
            with col_idx_1:
                st.caption(f"Период {year_range[0]}—{year_range[1]}")
                _render_index_progress(pd.DataFrame(proximity_weights.most_common(10), columns=["Лемма", "Индекс"]))
            with col_idx_2:
                st.caption(f"Период {year_range_2[0]}—{year_range_2[1]}")
                _render_index_progress(pd.DataFrame(proximity_weights_2.most_common(10), columns=["Лемма", "Индекс"]))

        with tab_delta:
            st.markdown("### 📈 Анализ изменений семантического поля")
            if delta_analysis is None:
                st.warning("Нет данных для дельта-анализа.")
            else:
                col_app, col_dis = st.columns(2)
                with col_app:
                    st.subheader("🟢 Топ появившихся слов")
                    if delta_analysis["appeared_words"]:
                        app_df = pd.DataFrame(delta_analysis["appeared_words"], columns=["Слово", "Индекс"])
                        app_df.index = range(1, len(app_df) + 1)
                        st.dataframe(app_df.head(10), width="stretch")
                    else:
                        st.info("Нет новых слов.")
                with col_dis:
                    st.subheader("🔴 Топ исчезнувших слов")
                    if delta_analysis["disappeared_words"]:
                        dis_df = pd.DataFrame(delta_analysis["disappeared_words"], columns=["Слово", "Индекс"])
                        dis_df.index = range(1, len(dis_df) + 1)
                        st.dataframe(dis_df.head(10), width="stretch")
                    else:
                        st.info("Нет исчезнувших слов.")

                st.divider()
                st.subheader("🔄 Самые существенные изменения индекса контекстуальной близости")
                if delta_analysis["changed_words"]:
                    changed_viz_data = []
                    for item in delta_analysis["changed_words"][:10]:
                        changed_viz_data.append(
                            {
                                "Слово": item["word"],
                                "Индекс период 1": f"{item['index_1']:.3f}",
                                "Индекс период 2": f"{item['index_2']:.3f}",
                                "Δ Индекс": f"{item['index_delta']:+.3f}",
                                "Δ %": f"{item['index_pct']:+.1f}%",
                                "Статус": "📈" if item["status"] == "growing" else ("📉" if item["status"] == "declining" else "➡️"),
                            }
                        )
                    changed_df = pd.DataFrame(changed_viz_data)
                    changed_df.index = range(1, len(changed_df) + 1)
                    st.dataframe(changed_df, width="stretch", hide_index=False)
                else:
                    st.info("Нет изменяющихся слов.")
        return

    tab_window, tab_index = st.tabs(["Классическое окно контекста", "Индекс Маяка"])
    with tab_window:
        data = top_neighbors["with_stopwords"] if count_stopwords else top_neighbors["filtered"]
        n_df = pd.DataFrame(data.most_common(10), columns=["Лемма", "Частота"])
        n_df.index = range(1, len(n_df) + 1)
        st.table(n_df)
    with tab_index:
        _render_index_progress(pd.DataFrame(proximity_weights.most_common(10), columns=["Лемма", "Индекс"]))


def _render_contexts(compare_periods, year_range, year_range_2, contexts, results_2):
    st.write("### Контексты употребления")

    if compare_periods and results_2:
        contexts_2 = results_2["contexts"]
        col_ctx_1, col_ctx_2 = st.columns(2)

        with col_ctx_1:
            st.subheader(f"Период {year_range[0]} — {year_range[1]} ({len(contexts)} контекстов)")
            if contexts:
                context_format = st.radio(
                    "Формат отображения (период 1):",
                    ["Таблица (базовая)", "Таблица (с выделением)"],
                    horizontal=True,
                    help="Выберите удобный способ просмотра контекстов",
                    key="ctx_fmt_1",
                )
                if context_format == "Таблица (базовая)":
                    display_contexts_table_simple(contexts)
                else:
                    display_contexts_table_highlighted(contexts)

        with col_ctx_2:
            st.subheader(f"Период {year_range_2[0]} — {year_range_2[1]} ({len(contexts_2)} контекстов)")
            if contexts_2:
                context_format_2 = st.radio(
                    "Формат отображения (период 2):",
                    ["Таблица (базовая)", "Таблица (с выделением)"],
                    horizontal=True,
                    help="Выберите удобный способ просмотра контекстов",
                    key="ctx_fmt_2",
                )
                if context_format_2 == "Таблица (базовая)":
                    display_contexts_table_simple(contexts_2)
                else:
                    display_contexts_table_highlighted(contexts_2)
            else:
                st.info("Контексты не найдены в этом периоде.")
        return

    if contexts:
        context_format = st.radio(
            "Формат отображения:",
            ["Таблица (базовая)", "Таблица (с выделением)"],
            horizontal=True,
            help="Выберите удобный способ просмотра контекстов",
        )
        if context_format == "Таблица (базовая)":
            display_contexts_table_simple(contexts)
        else:
            display_contexts_table_highlighted(contexts)


def render_word_tab(
    full_corpus,
    lemma_forms,
    search_word,
    year_range,
    compare_periods,
    year_range_2,
    window_size,
    count_stopwords,
    decay_distance,
    decay_brks,
    decay_sents,
    model_source,
    deepseek_key,
):
    if not search_word:
        return

    analysis_state = build_word_analysis_state(
        full_corpus=full_corpus,
        lemma_forms=lemma_forms,
        search_word=search_word,
        year_range=year_range,
        window_size=window_size,
        decay_distance=decay_distance,
        decay_brks=decay_brks,
        decay_sents=decay_sents,
        compare_periods=compare_periods,
        year_range_2=year_range_2,
    )

    if analysis_state is None:
        st.warning("Слово не найдено в корпусе.")
        st.stop()

    results = analysis_state["results"]
    results_2 = analysis_state["results_2"]

    if not results:
        st.warning("Слово не найдено в корпусе.")
        return

    st.markdown(f"## Анализ слова: `{analysis_state['target_word']}`")
    if compare_periods and year_range_2 is not None:
        st.caption(f"📊 Сравнение периодов: {year_range[0]} — {year_range[1]} vs {year_range_2[0]} — {year_range_2[1]}")
    else:
        st.caption(f"Период поиска: {year_range[0]} — {year_range[1]}")

    _render_synonyms(compare_periods, analysis_state["synonyms"], analysis_state["synonyms_filtered"])
    st.divider()
    _render_period_metrics(compare_periods, year_range, year_range_2, results, results_2, count_stopwords)
    st.divider()
    _render_semantic_connections(compare_periods, year_range, year_range_2, results, results_2, count_stopwords)
    _render_contexts(compare_periods, year_range, year_range_2, results["contexts"], results_2)
    render_llm_section(
        model_source=model_source,
        deepseek_key=deepseek_key,
        analysis_state=analysis_state,
        decay_distance=decay_distance,
        decay_brks=decay_brks,
        decay_sents=decay_sents,
    )
