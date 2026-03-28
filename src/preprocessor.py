import os
import re
import ast
import pandas as pd
import poetree # импортируем модуль для работы с корпусом PoeTree
import csv # импортируем модуль для работы с таблицами
from razdel import sentenize, tokenize
from pymorphy3 import MorphAnalyzer
from src.file_utils import get_files_in_folder, read_text_file, write_text_file, write_csv_file
from src.text_utils import morph, lemmatize, format_separate_poem, get_sentences

def import_poetree_corpora(author_id, author_name, directory, annual_limit=None, max_poems=None):
    '''
    Формирует корпус поэтических текстов автора в виде словаря на основании данных корпуса PoeTree.
    
    Рассчитана на работу с авторами русскоязычного корпуса.

    Args:
        author_id (int): Идентификатор поэта на Poetree.
        author_name (str): Имя автора, под которым стихотворения войдут в корпусную базу данных.
        directory (str): Путь, по которому будут сохранены файлы со всеми импортирвоанными корпусными стихотворениями.
        annual_limit (int): Лимит стихотворений одного года выпуска. Когда лимит достигнут, последующие стихотворения этого года пропускаются.
        max_poems (int): Общий лимит. Функция остановится, когда сохранит столько текстов.
    ''' 

    metadata_rows = []
    years_counter = {}

    i = 0
    processed_files_count = 0
    successfuly_processed = 0

    os.makedirs(directory, exist_ok=True)

    author = poetree.Author(lang='ru', id_=author_id)

    for poem in author.get_poems():

        if max_poems is not None and successfuly_processed >= max_poems:
            print('Достигунт лимит стихотворений! Завершаю импорт.')
            break

        try:
            metadata = poem.metadata()[0]
            if not metadata:
                raise ValueError('Ошибка! Не были получены метаданные')
            
            poem_title = metadata.get('title').replace('--', '—')
            poem_id = metadata.get('id_')
            year_created_to = metadata.get('year_created_to')

            if year_created_to:
                current_count = years_counter.get(year_created_to, 0)
                if annual_limit and current_count >= annual_limit:
                    print(f'Лимит за {year_created_to} год достигнут! Текст пропущен.')
                    continue
            else:
                print(f'Внимание! Год для стихотворения {poem_title} (id: {poem_id}) не указан. Поле года оставлено пустым.')
                # continue
            
            print(f'Начинаю обработку стихотворения {poem_title} (id: {poem_id})...')
            poem_body = poem.get_body()
            poem_lines = []
            last_stanza_id = None

            for line in poem_body:
                stanza_id = line['id_stanza']
                if last_stanza_id == None:
                    last_stanza_id = line['id_stanza']
                if stanza_id != last_stanza_id:
                    poem_lines.append('')
                    last_stanza_id = stanza_id
                poem_lines.append(line['text'])

            poem_text = '\n'.join(poem_lines).replace('--', '—')

            filename = f'{poem_id}.txt'

            with open(os.path.join(directory, filename), 'w', encoding='utf-8') as file:
                file.write(poem_text)
                
            row = {
                'filename': filename,
                'title': poem_title,
                'author': author_name,
                'year_finished': year_created_to,
                'genre': 'poetry'
            }
            metadata_rows.append(row)

            if year_created_to:
                years_counter[year_created_to] = years_counter.get(year_created_to, 0) + 1

            successfuly_processed += 1
            print(f'Стихотворение {poem_title} успешно обработано и добавлено в корпус. Метаданные сохранены')
        except Exception as e:
            print(f'Внимание, при обработке стихотворения {poem_title} (id: {poem_id}) возникла ошибка: {e}. Текст пропущен')
        finally:
            i += 1
            processed_files_count += 1
    
    headers = ['filename', 'title', 'author', 'year_finished', 'genre']
    filepath = os.path.join('data', 'metadata.csv')

    write_csv_file(filepath, metadata_rows, headers, rewrite=True)

    if processed_files_count == successfuly_processed:
        print(f'Все тексты были успешно введены в корпус, а их метаданные сохранены в таблицу!!!\nВсего обработано текстов: {processed_files_count}')
    else:
        print(f'''Обработка текстов закончена! Данные сохранены! Всего обработано текстов:
                {processed_files_count}, из них успешно {successfuly_processed}''')

def import_corpora_files(folder_path, corpus_path, genre='prose'):
    '''
    Прочитывает текстовые файлы, лежащие в папке назначения, и записывает их в корпус в обработанном виде вместе с метаданными.
    Может автоматически определять тип кодировки файла из четырёх наиболее популряных (utf-8, cp1251, ascii, latin-1)

    Args:
        folder_path (str): Исходная папка с необработанными текстами и файлом data.csv с именами файлов, заглавиями, указанием автора и года.
        corpus_path (str): Основная корпусная папка, в которую буду внесены обработанные тексты
        genre (str): Жанр, под которым стихотворения будут записаны в итоговую таблицу метаданных. По умолчанию указана проза.
    '''

    processed_files_count = 0
    successfuly_processed = 0
    metadata_rows = []
    rows_list = []

    files_list = get_files_in_folder(folder_path)
    metadata_path = os.path.join(folder_path, 'data.csv')

    encoding_types = ['"utf-8"', '"cp1251"', '"ascii"', '"latin-1"']
    
    with open(metadata_path, 'r', encoding='utf-8') as csv_file:
        metadata = csv.DictReader(csv_file)
        rows_list = list(metadata)
    
    for r in rows_list:
        filename = r['filename']
        filepath = os.path.join(folder_path, filename)
        print(f'Попытка прочитать файл {filename}...')
        content = None

        if filename in files_list:
            for tp in encoding_types:

                try:
                    with open(filepath, 'r', encoding=tp) as file:
                        content = file.read()
                    break
                except UnicodeDecodeError:
                    continue

            if content is not None:
                print(f"    Начало содержимого (прочитано): {content[:100]}...") 

                with open(os.path.join(corpus_path, genre, filename), 'w', encoding='utf-8') as file:
                        file.write(content)

                entry = {
                    'filename': filename,
                    'title': r['title'],
                    'author': r['author'],
                    'year_finished': r['year_finished'],
                    'genre': genre
                }

                metadata_rows.append(entry)
                successfuly_processed += 1
            else:
                print(f" 🚨 ОШИБКА: Не удалось прочитать файл '{filename}' ни одной из кодировок или файл пустой.")
            processed_files_count += 1

    headers = ['filename', 'title', 'author', 'year_finished', 'genre']
    filepath = os.path.join('data', 'metadata.csv')

    write_csv_file(filepath, metadata_rows, headers, rewrite=False)

    if processed_files_count == successfuly_processed:
        print(f'Все тексты были успешно введены в корпус, а их метаданные сохранены в таблицу!!!\nВсего обработано текстов: {processed_files_count}')
    else:
        print(f'''Обработка текстов закончена! Данные сохранены! Всего обработано текстов:
                {processed_files_count}, из них успешно {successfuly_processed}''')

# для каждого рода литературы свои правила обработки и соответственно своя отдельная функция

def process_poetry_corpus(raw_poetry_path, data_path, rewrite=True):
    '''
1. Читаем все тексты из папки, используя get_files_in_folder и read_text_file
2. Для каждого текста:
   - Разбиваем на предложения с помощью razdel.sentenize
   - Для каждого предложения:
     - Токенизируем его с помощью razdel.tokenize
     - Лемматизируем каждое слово, используя pymorphy3
     - Собираем леммы в структуру (список списков) для колонки 'lemmas'
   - Сохраняем оригинальный текст в колонке 'raw_text'
   - Форматируем текст для отображения (заменяем слэши на _BRK_ и добавляем переносы строк) и сохраняем в 'formatted_text'
   - Получаем метаданные (название, год, жанр) из metadata.csv по имени файла
3. Собираем все данные в DataFrame и сохраняем в database.csv
4. Дополнительно: функция для извлечения всего словаря автора и сохранения его в отдельный текстовый файл (author_vocabulary.txt)   
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
        
        # Разбиваем на предложения (используем razdel)

        formatted_text = format_separate_poem(text)
        
        sentences = get_sentences(formatted_text)
        
        all_tokens = []
        all_lemmas_separated = []
        all_lemmas_cleaned = []
        
        for sent in sentences:

            # Токенизируем каждое предложение
            tokens = list(tokenize(sent))

            sent_tokens = []
            sent_lemmas = []
            sent_lemmas_cleaned = []

            for token in tokens:

                token_text = token.text
                sent_tokens.append(token_text)

                # Сначала проверяем наш спецсимвол
                if token_text == '/':
                    sent_lemmas.append('_BRK_')
                    continue
                
                # Затем чистим всю остальную пунктуацию
                clean_token = re.sub(r'[^\w\s]', '', token_text)
                
                if not clean_token:
                    continue
                
                # Лемматизируем
                clean_token_lower = clean_token.lower()
                p = morph.parse(clean_token_lower)[0]
                sent_lemmas.append(p.normal_form)
                sent_lemmas_cleaned.append(p.normal_form)
            
            if sent_lemmas:
                all_tokens.append(sent_tokens)
                all_lemmas_separated.append(sent_lemmas)
                all_lemmas_cleaned.append(sent_lemmas_cleaned)
        
        row = metadata_df[metadata_df['filename'] == f]

        if not row.empty:
            text_metadata = row.iloc[0].to_dict() # Возвращает словарь со всеми полями (title, year, genre)
        else:
            text_metadata = {'title': 'Неизвестно', 'year_finished': 'Неизвестен', 'genre': 'Неизвестен'}

        data.append({
            'filename': f,
            'title': text_metadata.get('title'),
            'genre': text_metadata.get('genre'),
            'year_finished': text_metadata.get('year_finished'),
            'raw_text': text,
            'formatted_sentences': sentences,
            'tokens': all_tokens,
            'lemmas_separated': all_lemmas_separated,
            'lemmas_cleaned': all_lemmas_cleaned
        })

    df = pd.DataFrame(data)

    if rewrite:
        df.to_csv(database_path, mode='w', index=False, header=True, encoding='utf-8')
    else:
        df.to_csv(database_path, mode='a', index=False, header=not database_exists, encoding='utf-8')

    print(f'Обработка завершена УСПЕШНО! База сохранена в {database_path}')

    # Создаём словарь словоформ для оптимизации поиска контекстов
    print('Создаю словарь словоформ...')
    build_lemma_forms_mapping(database_path)

def save_author_vocabulary(database_path):
    """
    Извлекает все уникальные леммы из колонки 'lemmas_separated' и сохраняет в текстовый файл.
    """
    df = pd.read_csv(database_path)
    df['lemmas_separated'] = df['lemmas_separated'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

    vocab = set()

    for doc in df['lemmas_separated']:
        # doc — это список списков (предложений)
        for sentence in doc:
            for word in sentence:
                if word.isalpha(): # Берем только слова, игнорируем _BRK_
                    vocab.add(word.lower())

    vocab = sorted(list(vocab))

    vocab_path = os.path.join(os.path.dirname(database_path), 'author_vocabulary.txt')
    write_text_file(vocab_path, '\n'.join(vocab))

    return vocab

def build_lemma_forms_mapping(database_path):
    """
    Собирает словарь: лемма -> set всех встреченных словоформ в корпусе.
    Матчит tokens с lemmas_cleaned (без разделителей) для получения реальных форм.
    Сохраняет результат в JSON.
    """
    import json

    df = pd.read_csv(database_path)
    lemma_forms = {}

    for _, row in df.iterrows():
        try:
            tokens_sents = ast.literal_eval(row['tokens']) if pd.notnull(row['tokens']) else []
            lemmas_cleaned_sents = ast.literal_eval(row['lemmas_cleaned']) if pd.notnull(row['lemmas_cleaned']) else []

            # Матчим предложения
            for tokens, lemmas_clean in zip(tokens_sents, lemmas_cleaned_sents):
                token_idx = 0

                for lemma in lemmas_clean:
                    # Ищем соответствующий токен (пропускаем пунктуацию в tokens)
                    while token_idx < len(tokens):
                        token_text = tokens[token_idx]
                        token_idx += 1

                        # Очищаем от пунктуации
                        clean_token = re.sub(r'[^\w\s]', '', token_text)

                        if not clean_token:
                            continue  # Пропускаем чистую пунктуацию

                        # Добавляем словоform в набор
                        if lemma not in lemma_forms:
                            lemma_forms[lemma] = []

                        lemma_forms[lemma].append(clean_token.lower())
                        break

        except Exception as e:
            print(f"⚠️  Ошибка при обработке строки: {e}")
            continue

    # Преобразуем списки в sets для уникальности и сохраняем
    lemma_forms_unique = {lemma: list(set(forms)) for lemma, forms in lemma_forms.items()}

    forms_path = os.path.join(os.path.dirname(database_path), 'vocabulary_forms.json')
    with open(forms_path, 'w', encoding='utf-8') as f:
        json.dump(lemma_forms_unique, f, ensure_ascii=False, indent=2)

    print(f'✅ Словарь словоформ создан! Сохранено {len(lemma_forms_unique)} лемм в {forms_path}')
    return lemma_forms_unique

if __name__ == '__main__':
    process_poetry_corpus(rf'corpus/poetry', 'data')
    # save_author_vocabulary(rf'data/database.csv')