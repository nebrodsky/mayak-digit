import pandas as pd
import ast
from gensim.models import Word2Vec

# --- НА ДАННОМ ЭТАПЕ ИСПОЛЬЗОВАНИЕ МОДЕЛИ НЕ ПЛАНИРУЕТСЯ, ЭТО ПРОСТО СКРИПТ ДЛЯ ОБУЧЕНИЯ НОВОЙ МОДЕЛИ НА НАШЕМ КОРПУСЕ ---

# 1. Загрузка
print("Загрузка базы...")
df = pd.read_csv('data/database.csv')

# 2. Подготовка лемм (чистим от служебных символов)
def get_clean_sentences(df):
    sentences = []
    for l_str in df['lemmas']:
        try:
            nested = ast.literal_eval(l_str)
            # Убираем _BRK_ и пустые токены
            clean = [word for sublist in nested for word in sublist if not word.startswith('_')]
            if clean:
                sentences.append(clean)
        except:
            continue
    return sentences

print("Подготовка текстов...")
sentences = get_clean_sentences(df)

# 3. Обучение (Skip-gram, 100 эпох)
print("Обучение Word2Vec (это может занять минуту)...")
model = Word2Vec(
    sentences, 
    vector_size=100, 
    window=5, 
    min_count=4, # игнорируем совсем редкие опечатки
    sg=0,        # Skip-gram лучше для малых данных
    epochs=150   # Те самые 100 эпох для глубины
)

# 4. Сохранение
model.save("models/mayakovsky_v3.model")
print("Модель успешно сохранена в файл mayakovsky_v3.model!")