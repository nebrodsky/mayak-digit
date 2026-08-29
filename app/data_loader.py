import json
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from project_paths import DATA_DIR


load_dotenv()


@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_DIR / "database.parquet")
    return df.to_dict("records")


@st.cache_data
def load_lemma_forms():
    forms_path = DATA_DIR / "vocabulary_forms.json"

    if forms_path.exists():
        try:
            with open(forms_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Ошибка при загрузке словаря словоформ: {e}")
            return {}

    st.error(f"Файл {forms_path} не найден. Запустите препроцессинг.")
    return {}


@st.cache_data
def load_cluster_map():
    map_path = DATA_DIR / "word_cluster_map.json"
    if not map_path.exists():
        return None

    try:
        with open(map_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        return pd.DataFrame(records)
    except Exception as e:
        st.error(f"Ошибка при загрузке семантической карты: {e}")
        return None


@st.cache_data
def load_mayak_hapax():
    hapax_path = DATA_DIR / "mayak_hapax.json"
    if not hapax_path.exists():
        return None

    try:
        with open(hapax_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Ошибка при загрузке гапаксов: {e}")
        return None


def get_deepseek_key():
    return os.getenv("DEEPSEEK_API_KEY")
