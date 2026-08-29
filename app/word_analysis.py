import streamlit as st

from src.analyzer import filter_synonyms_by_corpus, full_word_analysis, get_unique_synonyms
from src.text_utils import russian_stopwords


def resolve_search_word(search_word, lemma_forms):
    normalized = search_word.strip().lower().replace("ё", "е")

    found_lemma = next(
        (lemma for lemma in lemma_forms if lemma.replace("ё", "е") == normalized),
        None,
    )
    if found_lemma:
        return found_lemma

    return next(
        (
            lemma
            for lemma, forms in lemma_forms.items()
            if any(form.replace("ё", "е") == normalized for form in forms)
        ),
        None,
    )


@st.cache_data
def cached_get_unique_synonyms(word, top_n=20, depth=50):
    return get_unique_synonyms(word, top_n_to_return=top_n, search_depth=depth)


@st.cache_data
def cached_full_word_analysis(
    _filtered_corpus,
    _lemma_forms,
    search_word,
    window_size,
    decay_distance,
    decay_brks,
    decay_sents,
):
    return full_word_analysis(
        filtered_corpus=_filtered_corpus,
        target_word=search_word,
        window_size=window_size,
        decay_distance=decay_distance,
        decay_brks=decay_brks,
        decay_sents=decay_sents,
        stopwords=russian_stopwords,
        lemma_forms=_lemma_forms,
    )


def build_word_analysis_state(
    full_corpus,
    lemma_forms,
    search_word,
    year_range,
    window_size,
    decay_distance,
    decay_brks,
    decay_sents,
    compare_periods=False,
    year_range_2=None,
):
    target_word = resolve_search_word(search_word, lemma_forms)
    if not target_word:
        return None

    filtered_corpus = [
        item for item in full_corpus
        if year_range[0] <= item["year_finished"] <= year_range[1]
    ]
    results = cached_full_word_analysis(
        filtered_corpus,
        lemma_forms,
        target_word,
        window_size,
        decay_distance,
        decay_brks,
        decay_sents,
    )

    results_2 = None
    if compare_periods and year_range_2 is not None:
        if year_range == year_range_2:
            results_2 = results
        else:
            results_2 = cached_full_word_analysis(
                [
                    item for item in full_corpus
                    if year_range_2[0] <= item["year_finished"] <= year_range_2[1]
                ],
                lemma_forms,
                target_word,
                window_size,
                decay_distance,
                decay_brks,
                decay_sents,
            )

    synonyms = cached_get_unique_synonyms(target_word, top_n=20, depth=50)
    synonyms_filtered = filter_synonyms_by_corpus(synonyms)

    return {
        "target_word": target_word,
        "filtered_corpus": filtered_corpus,
        "results": results,
        "results_2": results_2,
        "synonyms": synonyms,
        "synonyms_filtered": synonyms_filtered,
    }
