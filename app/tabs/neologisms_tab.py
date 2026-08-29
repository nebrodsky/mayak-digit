import pandas as pd
import streamlit as st

from corpus_metrics import check_hapax_in_navec, get_hapax_legomena
from data_loader import load_mayak_hapax


def render_neologisms_tab(full_corpus):
    st.markdown("## Неологизмы Маяковского")
    st.markdown("Слова, встречающиеся **только один раз** в его творчестве и **отсутствующие** в творчестве его современников (1900-1930)")

    st.info(
        "⚠️ **BETA версия**: Функционал анализа неологизмов находится в стадии разработки. "
        "В данный момент корректная фильтрация результатов ещё не реализована. "
        "Список будет дополняться дополнительными фильтрами и возможностями анализа."
    )

    hapax_data = load_mayak_hapax()
    if hapax_data is None:
        st.warning("❌ Файл с данными гапаксов не найден. ")
        return

    hapax_metadata = hapax_data.get("metadata", {})
    hapax_set = set(hapax_data.get("hapax_legomena", []))

    col1, col2 = st.columns(2)
    col1.metric("Произведений Маяка", hapax_metadata.get("mayakovsky_poems_count", 0))
    col2.metric("Уникальных лемм (Маяк)", f"{hapax_metadata.get('mayakovsky_unique_lemmas', 0):,}".replace(",", " "))

    st.divider()
    hapax_once = get_hapax_legomena(full_corpus, hapax_set)
    st.markdown(f"### Однократные единицы: {len(hapax_once):,}")

    if len(hapax_once) == 0:
        st.info("Нет однократных единиц!")
        return

    hapax_list = sorted(hapax_once.keys())
    _, not_in_navec = check_hapax_in_navec(hapax_list)

    tab_all_hapax, tab_unknown_vectors = st.tabs([
        f"Все однократные ({len(hapax_list)})",
        f"Отсутствуют в navec ({len(not_in_navec)})",
    ])

    with tab_all_hapax:
        hapax_df = pd.DataFrame({"Слово": hapax_list})
        hapax_df.index = range(1, len(hapax_df) + 1)
        st.dataframe(hapax_df, width="stretch", height=500)

        st.divider()
        csv = hapax_df.to_csv(index_label="№")
        st.download_button(
            label="Скачать список CSV",
            data=csv,
            file_name="mayak_hapax_legomena.csv",
            mime="text/csv",
        )

    with tab_unknown_vectors:
        st.markdown(
            "Эти слова встречаются только один раз у Маяковского "
            "и отсутствуют в векторной модели Navec (возможно, опечатки или уникальные авторские неологизмы)."
        )

        if len(not_in_navec) == 0:
            st.success("✅ Все однократные неологизмы представлены в модели navec")
        else:
            unknown_df = pd.DataFrame({"Слово": not_in_navec})
            unknown_df.index = range(1, len(unknown_df) + 1)
            st.dataframe(unknown_df, width="stretch", height=500)

            st.divider()
            csv_unknown = unknown_df.to_csv(index_label="№")
            st.download_button(
                label="📥 Скачать список CSV",
                data=csv_unknown,
                file_name="mayak_unknown_vectors.csv",
                mime="text/csv",
            )
