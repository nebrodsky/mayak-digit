import os
import requests
import poetree # импортируем модуль для работы с корпусом
import csv # импортируем модуль для работы с таблицами
import pandas as pd

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

def read_text_file(filepath, encoding='utf-8'):
    """
    Читает содержимое текстового файла.

    Args:
        filepath (str): Путь к файлу
        encoding (str): Тип кодировки. По умолчанию установлен utf-8.

    Returns:
        str: Содержимое файла или сообщение об ошибке
    """
    content = None
    filename = os.path.basename(filepath)

    try:
        with open(filepath, 'r', encoding=encoding) as f:
            content = f.read()
    except FileNotFoundError:
        print(f'Ошибка: Файл "{filename}" не найден')
    except UnicodeDecodeError:
        print(f'Ошибка: Неверная кодировка файла {filename}')
    return content

def read_csv_file(filepath):
    '''
    Считывает CSV-файл и преобразует его в список словарей.

    Args:
        filepath (str): Путь к CSV-файлу.

    Returns:
        list: Список словарей, где ключи — названия колонок. 
              Возвращает пустой список, если файл не найден.
    '''
    try:
        with open(filepath, 'r', encoding='utf-8') as csv_file:
            content = list(csv.DictReader(csv_file))
    except FileNotFoundError:
        print('ОШИБКА! Файл не найден.')
        content = []
    return content

def write_csv_file(filepath, data, headers, rewrite=True):
    '''
    Записывает данные в CSV файл.

    Args:
        filepath (str): Полный путь к файлу, включая папку и название файла
                       Например: 'results/statistics.csv'
        data (list): Список cловарей [{col1: val1}, {col2, value2}, ...]
        headers (list): Список заголовков ['col1', 'col2']
        rewrite (bool): Указывает, нужно ли перезаписывать файл, если он уже существует, или дополнить его

    Returns:
        bool: True если успешно
    '''
    folder = os.path.dirname(filepath)
    os.makedirs(folder, exist_ok=True) 
    
    try:
        with open(filepath, 'x', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)
        return True
    except FileExistsError:
        if rewrite:
            try:
                with open(filepath, 'w', newline='', encoding='utf-8') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(data)
                return True
            except FileNotFoundError:
                return False
        else:
            try:
                with open(filepath, 'a', newline='', encoding='utf-8') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=headers)
                    writer.writerows(data)
                return True
            except FileNotFoundError:
                return False
    except FileNotFoundError:
        return False

def write_text_file(filepath, content, rewrite=True):
    '''
    Записывает текст в файл, автоматически создавая необходимые директории.

    Args:
        filepath (str): Путь к файлу (например, 'results/output.txt').
        content (str): Текст для записи.
        rewrite (bool): Если True — перезаписывает файл, если False — добавляет текст в конец.

    Returns:
        bool: True, если запись прошла успешно, False — если возникла ошибка.
    '''
    folder = os.path.dirname(filepath)
    os.makedirs(folder, exist_ok=True)

    try:
        with open(filepath, 'x', newline='', encoding='utf-8') as f:
            f.write(content)
        return True
    except FileExistsError:
        if rewrite:
            try:
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    f.write(content)
                    return True
            except FileNotFoundError:
                return False
        else:
            try:
                with open(filepath, 'a', newline='', encoding='utf-8') as f:
                    f.write(content)
                return True
            except FileNotFoundError:
                return False
    except FileNotFoundError:
        return False

def get_files_in_folder(folder_path, extension='.txt'):
    """
    Получает список файлов в указанной папке с заданным расширением.

    Args:
        folder_path (str): Путь к папке
        extension (str): Расширение файлов (по умолчанию '.txt')

    Returns:
        list: Список имен файлов с указанным расширением
    """
    files_list = []

    for filename in os.listdir(folder_path):
        if filename.endswith(extension):
            files_list.append(filename)

    return files_list

def get_text_metadata(filename, metadata_path):
    '''
    Docstring для get_text_metadata
    
    :param filename: Описание
    :param metadata_path: Описание
    '''

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
    else:
        print(f'ОШИБКА! Файл с исходными метаданными не найден')
        return None

    row = metadata_df[metadata_df['filename'] == filename]

    if not row.empty:
        return row.iloc[0].to_dict() # Возвращает словарь со всеми полями (title, year, genre)
    return {"title": "Unknown", "year": "Unknown", "genre": "Unknown"}

if __name__ == "__main__":
    import_poetree_corpora(18, "Маяковский", os.path.join('corpus', 'poetry'))
    # import_corpora_files('raw_entries', 'corpus')