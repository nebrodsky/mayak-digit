import os
import re
import pymorphy3
from razdel import sentenize, tokenize
from src.file_utils import read_text_file
from pymystem3 import Mystem

morph = pymorphy3.MorphAnalyzer()
ms = Mystem(disambiguation=True, grammar_info=True, entire_input=False)
ms.start()

russian_stopwords = set("""
    и в во не что он на я с со как а то все она так его но да ты к у же вы за б бы по
    только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли
    если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя
    ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без
    будто чего раз тоже себе под будет ж тогда кто это этот того потому этого какой
    совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех
    никогда можно при наконец два об другой хоть после над больше тот через эти
    нас про всего них какая много разве три эту моя впрочем хорошо свою этой
    перед иногда лучше чуть том нельзя такой им более всегда конечно всю между
""".split())

# --- ФУНКЦИИ ОЧИСТКИ ТЕКСТА ---

def clean_punctuation(text):
    """Удаляет знаки пунктуации из текста, заменяя их на пробелы."""
    punct = '\'\",.!?;:—-""«»()[]/'
    clean_text = text.lower()
    for char in punct:
        clean_text = clean_text.replace(char, ' ')
    return clean_text

def clean_mayakovsky_fragmentation(text):
    """
    Нормализует специфику записи Маяковского:
    - склеивает переносы строк внутри слова (сло- / во → слово /)
    - убирает акцентные дефисы (н-а-ш-е → наше)
    """
    text = re.sub(r'(\w+)-\s*/\s*(\w+)', r'\1\2 /', text)
    text = re.sub(r'(\b\w)(-\w)+\b', lambda m: m.group(0).replace('-', ''), text)
    return text

# --- ПРОСТЫЕ СТАТИСТИЧЕСКИЕ ФУНКЦИИ ---

def count_words(text, remove_punct=True):
    """Подсчитывает количество слов в тексте."""
    clean_text = clean_punctuation(text) if remove_punct else text
    return len(clean_text.split())

# --- ТОКЕНИЗАЦИЯ И ЛЕММАТИЗАЦИЯ ---

def tokenizator(text, remove_punct=False):
    """Разбивает текст на токены с помощью razdel."""
    tokens = list(tokenize(text))
    if remove_punct:
        for token in tokens:
            token.text = clean_punctuation(token.text)
    return tokens

def lemmatize(text, remove_punct=True):
    """Лемматизирует текст через pymorphy3."""
    clean_text = clean_punctuation(text) if remove_punct else text
    words = clean_text.split()
    return [morph.parse(word)[0].normal_form for word in words if word]

def lemmatize_with_mystem(sentence):
    """
    Лемматизирует предложение: razdel для токенизации, MyStem для лемм и POS-тегов.
    Обрабатывает спецсимволы лесенки Маяковского (_BRK_), цифровые токены и дефисные слова.

    Возвращает словарь:
        tokens            — список токенов (razdel)
        lemmas            — список лемм (включая _BRK_ маркеры)
        lemmas_clean      — леммы без _BRK_
        pos_tags          — POS-теги
        lemmas_pos_tagged — строки "лемма/POS"
    """
    sentence_cleaned = clean_mayakovsky_fragmentation(sentence)

    tokens = list(tokenize(sentence_cleaned))
    tokens = [token.text for token in tokens]

    sentence_mystem = sentence_cleaned.replace('/', ' ')
    ms_analysis = ms.analyze(sentence_mystem)
    ms_words = [
        word for word in ms_analysis
        if word['text'].strip() and any(c.isalpha() for c in word['text'])
    ]

    lemmas = []
    lemmas_clean = []
    pos_tags = []
    lemmas_pos = []

    last_mystm_pos = 0

    for tk in tokens:
        token_text = tk.lower()

        if token_text == '/':
            lemmas.append('_BRK_')
            continue

        if not any(c.isalnum() for c in token_text):
            continue

        if any(c.isdigit() for c in token_text):
            lemmas.append(token_text)
            lemmas_clean.append(token_text)
            pos_tags.append('NUM')
            lemmas_pos.append(f"{token_text}/NUM")
            continue

        if '-' in token_text:
            ms_result = ms.analyze(token_text)
            if len(ms_result) == 1 and ms_result[0]['analysis']:
                info = ms_result[0]['analysis'][0]
                lemma = info['lex']
                pos = info.get('gr', '').split(',')[0].split('=')[0]
            elif len(ms_result) == 2 and any(info['analysis'] for info in ms_result):
                token_text_parts = token_text.split('-')
                lemma_parts = []
                for indx, res in enumerate(ms_result):
                    part_result = ms_result[indx]['analysis']
                    if not part_result:
                        lemma_parts.append(token_text_parts[indx])
                        print(f"⚠️ Не удалось лемматизировать часть дефисного слова '{token_text_parts[indx]}' в '{token_text}'. Добавляем как есть.")
                        continue
                    lemma_parts.append(part_result[0]['lex'])
                lemma = '-'.join(lemma_parts)
                pos = 'HYPHCOMP'
            else:
                lemma = token_text
                pos = 'UNK'
            lemmas.append(lemma)
            lemmas_clean.append(lemma)
            pos_tags.append(pos)
            lemmas_pos.append(f"{lemma}/{pos}")
            continue

        while last_mystm_pos < len(ms_words):
            ms_word = ms_words[last_mystm_pos]
            ms_initial = ms_word['text']

            if ms_initial == tk:
                if ms_word['analysis']:
                    info = ms_word['analysis'][0]
                    lemma = info['lex']
                    pos = info.get('gr', '').split(',')[0].split('=')[0]
                else:
                    lemma = ms_initial
                    pos = 'UNK'

                lemmas.append(lemma)
                lemmas_clean.append(lemma)
                pos_tags.append(pos)
                lemmas_pos.append(f"{lemma}/{pos}")

                last_mystm_pos += 1
                break
            else:
                last_mystm_pos += 1

    if last_mystm_pos != len(ms_words):
        print(
            f"⚠️ Не все слова обработаны MyStem! "
            f"Обработано {last_mystm_pos} из {len(ms_words)}. "
            f"Необработанные: {[w['text'] for w in ms_words[last_mystm_pos:]]}"
        )

    return {
        'tokens': tokens,
        'lemmas': lemmas,
        'lemmas_clean': lemmas_clean,
        'pos_tags': pos_tags,
        'lemmas_pos_tagged': lemmas_pos
    }

# --- РАБОТА С ТЕКСТОМ ---

def get_sentences(text):
    """Разбивает текст на предложения с помощью razdel."""
    return [sentence.text.replace('\n', ' ') for sentence in sentenize(text)]

def format_separate_poem(text):
    """
    Форматирует стихотворение для хранения в корпусе:
    строфы разделяются ' // ', строки — ' / '.
    """
    formatted_text = text.strip()
    formatted_text = formatted_text.replace('\n\n', ' // ').replace('\n', ' / ')
    return formatted_text
