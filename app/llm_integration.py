import streamlit as st

from src.analyzer import (
    prepare_llm_prompt,
    prompt_prefix,
    proximity_neighbours_for_synonyms,
    synonyms_proximity_index,
)
from src.text_utils import russian_stopwords


def render_llm_section(
    model_source,
    deepseek_key,
    analysis_state,
    decay_distance,
    decay_brks,
    decay_sents,
):
    results = analysis_state["results"]
    if not results:
        return

    if not st.button("Запустить анализ через LLM"):
        return

    status_text = st.empty()

    with st.spinner("Собираем статистику для промпта... Пожалуйста, подождите."):
        status_text.text("📊 Рассчитываем семантическую близость синонимов...")
        syn_prox_index = synonyms_proximity_index(
            analysis_state["target_word"],
            analysis_state["synonyms_filtered"],
            results["proximity_weights"],
        )

        status_text.text("Считаем индекс Маяка для синонимов (это может занять время)...")
        neighbors_for_syns = proximity_neighbours_for_synonyms(
            analysis_state["synonyms_filtered"],
            analysis_state["filtered_corpus"],
            decay_distance,
            decay_brks,
            decay_sents,
            stopwords=russian_stopwords,
        )

        status_text.text("Формирую аналитическое досье для ИИ...")
        interpr_prompt = prepare_llm_prompt(
            target_word=analysis_state["target_word"],
            synonyms=analysis_state["synonyms"],
            synonyms_filtered=analysis_state["synonyms_filtered"],
            syn_proximity=syn_prox_index,
            neighbors_for_synonyms=neighbors_for_syns,
            total_occurrences=results["total_occurrences"],
            year_dist=results["year_dist"],
            proximity_weights=results["proximity_weights"],
        )
        status_text.empty()

    st.subheader("Сгенерированный промпт для ИИ:")
    st.code(interpr_prompt, language="text")
    st.divider()
    st.subheader("Аналитический комментарий от LLM:")
    st.info("Несмотря на предварительную настройку, LLM может добавлять к реальным данным собственные интерпретации. Пожалуйста, относитесь к результату критически и сверяйтесь с фактическими данными из предыдущих разделов.")

    if model_source == "Локальная (Ollama)":
        import ollama

        response_container = st.empty()
        ollama_prompt = prompt_prefix + "\n\n" + interpr_prompt
        full_response = ""

        try:
            stream = ollama.generate(model="llama3:8b", prompt=ollama_prompt, stream=True)
            for chunk in stream:
                full_response += chunk["response"]
                response_container.markdown(full_response + "▌")
            response_container.markdown(full_response)
        except Exception as e:
            st.error(f"Ошибка при обращении к Ollama: {e}")
            st.info("Убедитесь, что приложение Ollama запущено и модель llama3:8b скачана.")

    elif model_source == "DeepSeek 4 (API)":
        if not deepseek_key:
            st.error("Ключ DeepSeek не найден в .env! Добавьте DEEPSEEK_API_KEY.")
            return

        from openai import OpenAI as DeepSeekClient

        client_ds = DeepSeekClient(api_key=deepseek_key, base_url="https://api.deepseek.com")

        with st.spinner("DeepSeek анализирует семантические поля..."):
            try:
                response = client_ds.chat.completions.create(
                    model="deepseek-v4-flash",
                    messages=[
                        {"role": "system", "content": prompt_prefix},
                        {"role": "user", "content": interpr_prompt},
                    ],
                    stream=False,
                    extra_body={"thinking": {"type": "enabled"}},
                )
                st.markdown(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Ошибка API DeepSeek: {e}")
                st.info("Убедитесь, что ключ DEEPSEEK_API_KEY корректно настроен.")
