import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
import numpy as np

# 1. Загрузка модели
model = Word2Vec.load("models/mayak_w2v_skipgram.model")

# 2. Выбор слов для визуализации
# Возьмем топ-100 слов по частоте + наш тег
words = list(model.wv.index_to_key[:100])
if "_BRK_" not in words:
    words.append("_BRK_")

# Получаем векторы этих слов
word_vectors = np.array([model.wv[w] for w in words])

# 3. Снижение размерности (t-SNE)
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(words)-1))
vectors_2d = tsne.fit_transform(word_vectors)

# 4. Отрисовка
plt.figure(figsize=(15, 10))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.5, c='red')

# Добавляем подписи
for i, word in enumerate(words):
    # Выделим наш тег цветом или жирным шрифтом
    color = 'blue' if word == '_BRK_' else 'black'
    weight = 'bold' if word == '_BRK_' else 'normal'
    
    plt.annotate(word, 
                 xy=(vectors_2d[i, 0], vectors_2d[i, 1]), 
                 xytext=(5, 2), 
                 textcoords='offset points', 
                 ha='right', 
                 va='bottom',
                 color=color,
                 weight=weight)

plt.title("Семантическая карта мира Маяковского (Word2Vec + t-SNE)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()