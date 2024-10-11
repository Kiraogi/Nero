import pandas as pd
import dask.dataframe as dd
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
import streamlit as st
import io
import torch
import os
import re
from datetime import datetime

"""
Задачи на ближайшие время:
1) Доработать модель: 
    ∟Вывод top3 совпадений (выбор сколько выводить)
    ∟Добавить возможность выставления коффициента 
    ∟Добавить проверку на Англ языке 
    ∟Дообучить
2) Добавить кнопки для выбора модели, сохранения ее
"""

model_directory = './models/'  # Путь к директории для хранения моделей


# Функция для получения следующего имени модели с увеличением версии
def get_next_model_name():
    current_date = datetime.now().strftime("%d.%m.%Y")
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    model_files = [f for f in os.listdir(model_directory) if re.match(
        r'^\d{2}\.\d{2}\.\d{4} Nero V\d+\.\d+$', f)]
    if not model_files:
        return f"{current_date} Nero V1.0"
    latest_model = sorted(model_files)[-1]
    version_match = re.search(r'V(\d+)\.(\d+)$', latest_model)
    if version_match:
        major, minor = map(int, version_match.groups())
        new_version = f"V{major}.{minor + 1}"
        return f"{current_date} Nero {new_version}"
    return f"{current_date} Nero V1.0"

# Выбор новой модели
main_model = SentenceTransformer('gtr-t5-large')  # Более продвинутая модель
fine_tuned_model = None

# Загрузка модели (добавьте в соответствующее место кода)
def load_model(path):
    model = SentenceTransformer('gtr-t5-large')  # Используем более мощную модель
    try:
        model.load_state_dict(torch.load(path))
        st.write("Загружена дообученная модель.")
    except FileNotFoundError:
        st.write("Файл модели не найден. Загружена предобученная модель.")
    except Exception as e:
        st.write(f"Произошла ошибка при загрузке модели: {e}")
    return model

# Функция для сохранения модели
def save_model(model, path):
    torch.save(model.state_dict(), path)
    st.write(f"Модель сохранена как: {path}")

# Функция для загрузки данных
def load_data(file):
    try:
        data = pd.read_excel(file, engine='openpyxl')
        return data
    except Exception as e:
        st.write(f"Ошибка при загрузке файла: {e}")
        return None

# Основное окно
model_path = os.path.join(model_directory, get_next_model_name())
st.title("Неро")
st.write("Система для поиска совпадений между названиями товаров")
# Загрузка данных
main_model = SentenceTransformer('gtr-t5-large')
fine_tuned_model = None

if st.button("Дообучить модель", key="fine_tune_model"):
    fine_tuned_model = main_model
    st.write("Модель дообучена.")

if st.button("Сохранить модель", key="save_model"):
    if fine_tuned_model:
        save_model(fine_tuned_model, model_path)
    else:
        st.write("Нет дообученной модели для сохранения.")

model_files = sorted(os.listdir(model_directory))
selected_model = st.selectbox("Выберите модель", model_files)

if selected_model:
    model = load_model(os.path.join(model_directory, selected_model))
    st.write(f"Используется модель: {selected_model}")

# Функция для вычисления эмбеддингов
@st.cache_data
def compute_embeddings(names, _model):
    return _model.encode(names, convert_to_tensor=True)

# Функция для обучения
def train_model(model, examples, new_data=None):
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=16)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    if new_data is not None:
        new_examples = [InputExample(texts=[new_data[i]], label=1)
                        for i in range(len(new_data))]
        new_dataloader = DataLoader(new_examples, shuffle=True, batch_size=16)
        model.fit(train_objectives=[
                  (train_dataloader, train_loss), (new_dataloader, train_loss)], epochs=1)
    else:
        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1)
    save_model(model, model_path)

# Функция для поиска совпадений
def find_matching_names(our_product_names, competitor_product_names, threshold=0.8, model=None):
    if model is None:
        raise ValueError("Модель не передана")

    main_embeddings_our = compute_embeddings(our_product_names, model)
    main_embeddings_competitor = compute_embeddings(
        competitor_product_names, model)

    if fine_tuned_model:
        fine_tuned_embeddings_our = compute_embeddings(
            our_product_names, fine_tuned_model)
        fine_tuned_embeddings_competitor = compute_embeddings(
            competitor_product_names, fine_tuned_model)
        similarities = (util.pytorch_cos_sim(main_embeddings_our, main_embeddings_competitor) +
                        util.pytorch_cos_sim(fine_tuned_embeddings_our, fine_tuned_embeddings_competitor)) / 2
    else:
        similarities = util.pytorch_cos_sim(
            main_embeddings_our, main_embeddings_competitor)

    matching_names = []
    for i, our_name in enumerate(our_product_names):
        for j, competitor_name in enumerate(competitor_product_names):
            if similarities[i][j] >= threshold:
                matching_names.append(
                    (our_name, competitor_name, similarities[i][j].item()))

    matching_names = sorted(matching_names, key=lambda x: (-x[2], x[0]))

    return matching_names


# Интерфейс для загрузки файлов и выбора колонок
st.title('Сравнение названий товаров')
st.write('Загрузите файлы с названиями ваших товаров и товаров конкурента для поиска совпадений.')

our_file = st.file_uploader('Загрузите файл с вашими товарами', type=['xlsx'])
competitor_file = st.file_uploader(
    'Загрузите файл с товарами конкурента', type=['xlsx'])
examples_file = st.file_uploader(
    'Загрузите файл с примерами соответствий (необязательно)', type=['xlsx'])

# Переменные для данных и результатов
our_data, competitor_data = None, None
results_df = pd.DataFrame()
# Функция для вычисления совпадений
def run_computation(our_data, our_column, competitor_data, competitor_column, examples_data=None, examples_column_our=None, examples_column_competitor=None):
    if our_data is None or competitor_data is None:
        st.write("Не удалось загрузить данные. Проверьте файлы и попробуйте снова.")
        return

    our_product_names = our_data[our_column].tolist()
    competitor_product_names = competitor_data[competitor_column].tolist()

    examples = []
    if examples_data is not None and examples_column_our and examples_column_competitor:
        for index, row in examples_data.iterrows():
            example = InputExample(
                texts=[row[examples_column_our], row[examples_column_competitor]])
            examples.append(example)
        st.write('Дообучение модели на примерах...')
        train_model(fine_tuned_model, examples)

    st.write('Вычисление эмбеддингов и поиск совпадений...')
    matching_names = find_matching_names(
        our_product_names, competitor_product_names, threshold=0.8, model=main_model)

    global results_df  # Используем глобальную переменную для хранения результатов
    if matching_names:
        results_df = pd.DataFrame(matching_names, columns=[
                                  'Наше название', 'Название конкурента', 'Схожесть'])
        st.write('Все совпадения:')
        st.dataframe(results_df)
    else:
        st.write('Совпадений не найдено.')


# Кнопка для запуска вычислений
if st.button("Начать вычисления", key="run_computation"):
    if our_file and competitor_file:
        our_data = load_data(our_file)
        competitor_data = load_data(competitor_file)

        if our_data is not None and competitor_data is not None:
            our_columns = our_data.columns.tolist()
            competitor_columns = competitor_data.columns.tolist()

            our_column = st.selectbox(
                'Выберите колонку с названиями ваших товаров', our_columns, key='our_column')
            competitor_column = st.selectbox(
                'Выберите колонку с названиями товаров конкурента', competitor_columns, key='competitor_column')

            if examples_file:
                examples_data = load_data(examples_file)
                if examples_data is not None:
                    examples_column_our = st.selectbox(
                        'Выберите колонку с вашими названиями из примеров', examples_data.columns.tolist(), key='examples_column_our')
                    examples_column_competitor = st.selectbox(
                        'Выберите колонку с названиями конкурента из примеров', examples_data.columns.tolist(), key='examples_column_competitor')
                    run_computation(our_data, our_column, competitor_data, competitor_column,
                                    examples_data, examples_column_our, examples_column_competitor)
            else:
                run_computation(our_data, our_column,
                                competitor_data, competitor_column)

# Кнопка для скачивания результатов
if not results_df.empty:
    st.write("Скачайте результаты вычислений:")
    output = io.BytesIO()
    
    # Сохранение результатов в Excel и запись в буфер
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        results_df.to_excel(writer, index=False, sheet_name='Results')
        writer.book.close()  # Закрываем Workbook явно

    # Получаем данные из буфера и сбрасываем его позицию на начало
    output.seek(0)
    processed_data = output.read()

    st.download_button(
        label="Скачать Excel",
        data=processed_data,
        file_name='results.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

def multiply(a, b=4, c):
    return a * b ** c
result = multiply(5, 3, 2)