import os
import re
import pandas as pd
from razdel import sentenize, tokenize
from pymorphy3 import MorphAnalyzer
from file_utils import get_files_in_folder, read_text_file
from text_utils import lemmatize, separate_poem

morph = MorphAnalyzer()

# для каждого рода литературы свои правила обработки и соответственно своя отдельная функция

def process_poetry_corpus(raw_poetry_path, data_path):
    '''
    Docstring для process_corpus
    
    :param raw_corpus_path: Описание
    :param processed_corpus_path: Описание
    '''
    data = []

    metadata_path = os.path.join(data_path, 'metadata.csv')
    database_path = os.path.join(data_path, 'database.csv')
    database_exists = os.path.isfile(database_path)

    files_list = get_files_in_folder(raw_poetry_path)
    
    if files_list:
        print(f'Проверка папки с текстами завершена УСПЕШНО! Найдено файлов: {len(files_list)}')
    else:
        print(f'ОШИБКА! В исходной папке не найдено ни одного текста')
        return None
    
    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
    else:
        print(f'ОШИБКА! Файл с исходными метаданными не найден')
        return None

    for f in files_list:

        text = read_text_file(os.path.join(raw_poetry_path, f))

        # 1. ПРЕПРОЦЕССИНГ И ТОКЕНИЗАЦИЯ
        # Заменяем слэши на наш технический токен
        # prepared_text = text.replace(' / ', ' _BRK_ ')
        
        # Разбиваем на предложения (используем razdel)
        sentences = list(sentenize(text))
        
        all_lemmas_structured = []
        
        for sent in sentences:
            # Токенизируем каждое предложение
            tokens = [t.text for t in tokenize(sent.text)]

            sent_lemmas = []
            for token in tokens:

                if token == '/':
                    sent_lemmas.append('_BRK_')

                # Очищаем от пунктуации, но сохраняем наш токен
                clean_token = re.sub(r'[^\w\s_]', '', token)
                
                if not clean_token:
                    continue
                
                # Приводим к нижнему регистру для сравнения
                clean_token_lower = clean_token.lower()
                
                if clean_token_lower == '_brk_':
                    sent_lemmas.append('_BRK_')
                else:
                    # Лемматизируем обычные слова
                    p = morph.parse(clean_token_lower)[0]
                    sent_lemmas.append(p.normal_form)
            
            if sent_lemmas:
                all_lemmas_structured.append(sent_lemmas)
        
        row = metadata_df[metadata_df['filename'] == f]

        if not row.empty:
            text_metadata = row.iloc[0].to_dict() # Возвращает словарь со всеми полями (title, year, genre)
        else:
            text_metadata = {'title': 'Неизвестно', 'year_finished': 'Неизвестен', 'genre': 'Неизвестен'}

        data.append({
            'filename': f,
            'text_name': text_metadata.get('title'),
            'genre': text_metadata.get('genre'),
            'year_finished': text_metadata.get('year_finished'),
            'raw_text': text,
            'formatted_text': separate_poem(text),
            'lemmas': all_lemmas_structured
        })

    df = pd.DataFrame(data)

    df.to_csv(database_path, mode='a', index=False, header=not database_exists, encoding='utf-8')

    print(f'Обработка завершена УСПЕШНО! База сохранена в {database_path}')

if __name__ == '__main__':
    process_poetry_corpus(rf'corpus/poetry', 'data')