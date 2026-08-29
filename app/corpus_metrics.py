from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import streamlit as st

from src.analyzer import navec
from src.text_utils import russian_stopwords


@st.cache_data
def get_hapax_legomena(corpus_records, hapax_set):
    freq_counter = Counter()
    for item in corpus_records:
        for sentence in item["lemmas_cleaned"]:
            freq_counter.update(sentence)

    return {
        lemma: 1
        for lemma in hapax_set
        if freq_counter.get(lemma, 0) == 1 and not any(c.isdigit() for c in lemma)
    }


@st.cache_data
def check_hapax_in_navec(hapax_list):
    in_navec = []
    not_in_navec = []
    unk_id = navec.vocab.unk_id

    for lemma in hapax_list:
        word_id = navec.vocab.get(lemma, unk_id)
        if word_id != unk_id:
            in_navec.append(lemma)
        else:
            not_in_navec.append(lemma)

    return in_navec, not_in_navec


@st.cache_data
def compute_general_metrics(corpus_records):
    total_texts = len(corpus_records)
    total_tokens = sum(len(item["tokens"]) for item in corpus_records)
    all_lemmas = []

    for text in corpus_records:
        for sentence in text["lemmas_cleaned"]:
            all_lemmas.extend(sentence)

    total_lemmas = len(all_lemmas)
    unique_lemmas = len(set(all_lemmas))
    texts_by_year = Counter(item["year_finished"] for item in corpus_records)

    return {
        "total_texts": total_texts,
        "total_tokens": total_tokens,
        "total_lemmas": total_lemmas,
        "unique_lemmas": unique_lemmas,
        "texts_by_year": dict(texts_by_year),
    }


@st.cache_data
def compute_frequency_dict(corpus_records, exclude_stopwords=True):
    counter = Counter()
    for item in corpus_records:
        for sentence in item["lemmas_cleaned"]:
            counter.update(sentence)

    if exclude_stopwords:
        for stopword in russian_stopwords:
            counter.pop(stopword, None)

    return counter


@st.cache_data
def compute_vocabulary_growth(corpus_records):
    by_year = defaultdict(list)
    for item in corpus_records:
        for sentence in item["lemmas_cleaned"]:
            by_year[item["year_finished"]].extend(sentence)

    sorted_years = sorted(by_year.keys())
    seen_lemmas = set()
    growth_data = []

    for year in sorted_years:
        lemmas_year = by_year[year]
        new_lemmas_count = len(set(lemmas_year) - seen_lemmas)
        seen_lemmas.update(lemmas_year)
        unique_in_year = len(set(lemmas_year))
        total_in_year = len(lemmas_year)
        growth_data.append(
            {
                "Год": year,
                "Уникальных лемм накоплено": len(seen_lemmas),
                "Новых лемм": new_lemmas_count,
                "Всего лемм в году": total_in_year,
                "Type-Token Ratio": round(unique_in_year / total_in_year, 3) if total_in_year > 0 else 0,
            }
        )

    return growth_data


@st.cache_data
def compute_vector_map(corpus_records, top_n=100, exclude_stopwords=True, pca_base_size=300):
    if pca_base_size < top_n:
        pca_base_size = top_n

    freq_counter = compute_frequency_dict(corpus_records, exclude_stopwords=exclude_stopwords)
    all_words_for_pca = [word for word, _ in freq_counter.most_common(pca_base_size)]

    words_in_navec = [(word, freq_counter[word]) for word in all_words_for_pca if word in navec]
    if len(words_in_navec) < 3:
        return None

    words, freqs = zip(*words_in_navec)
    matrix = np.array([navec[word] for word in words])
    centered = matrix - matrix.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T

    return pd.DataFrame(
        {
            "Слово": list(words)[:top_n],
            "x": coords[:top_n, 0],
            "y": coords[:top_n, 1],
            "Частота": list(freqs)[:top_n],
        }
    )
