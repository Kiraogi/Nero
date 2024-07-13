import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transliterate import translit
import streamlit as st

# Загрузка предобученной модели Sentence-BERT
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


# Функции для предобработки данных
@st.cache
def preprocess_names(names):
    preprocessed_names = []
    for name in names:
        # Транслитерация с английского на русский
        transliterated_name = translit(name, 'ru', reversed=True)
        # Очистка и приведение к нижнему регистру
        preprocessed_name = transliterated_name.lower().strip()
        preprocessed_names.append(preprocessed_name)
    return preprocessed_names


# Функция для загрузки данных
@st.cache
def load_data(file):
    return pd.read_excel(file)


# Функция для вычисления эмбеддингов
@st.cache(allow_output_mutation=True)
def compute_embeddings(names):
    return model.encode(names, convert_to_tensor=True)


# Функция для нахождения совпадающих названий
def find_matching_names(our_product_names, competitor_product_names, threshold=0.8):
    our_embeddings = compute_embeddings(our_product_names)
    competitor_embeddings = compute_embeddings(competitor_product_names)

    similarities = util.pytorch_cos_sim(our_embeddings, competitor_embeddings)

    matching_names = []
    for i, our_name in enumerate(our_product_names):
        for j, competitor_name in enumerate(competitor_product_names):
            if similarities[i][j] >= threshold:
                matching_names.append((our_name, competitor_name, similarities[i][j].item()))

    return matching_names


# Создание визуального интерфейса с использованием Streamlit
st.title('Сравнение названий товаров')
st.write('Загрузите файлы с названиями ваших товаров и товаров конкурента для поиска совпадений.')

our_file = st.file_uploader('Загрузите файл с вашими товарами', type=['xlsx'])
competitor_file = st.file_uploader('Загрузите файл с товарами конкурента', type=['xlsx'])

if our_file and competitor_file:
    our_data = load_data(our_file)
    competitor_data = load_data(competitor_file)

    our_product_names = preprocess_names(our_data['ProductName'].tolist())
    competitor_product_names = preprocess_names(competitor_data['ProductName'].tolist())

    matching_names = find_matching_names(our_product_names, competitor_product_names)

    if matching_names:
        results_df = pd.DataFrame(matching_names, columns=['Наше название', 'Название конкурента', 'Схожесть'])
        st.write('Найденные совпадения:')
        st.dataframe(results_df)

        results_file = st.download_button(
            label="Скачать результаты",
            data=results_df.to_excel(index=False),
            file_name='matching_products.xlsx'
        )
    else:
        st.write('Совпадений не найдено.')
