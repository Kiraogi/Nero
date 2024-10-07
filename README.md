# Проект "Неро" - Система для поиска совпадений между названиями товаров

Проект "Неро" представляет собой систему, предназначенную для поиска совпадений между названиями товаров на основе эмбеддингов. Система использует предобученную модель GTR-T5 для генерации эмбеддингов и поиска совпадений.

## Функционал
* Загрузка данных из Excel-файлов
* Выбор колонок с названиями товаров
* Дообучение модели на примерах
* Вычисление эмбеддингов и поиск совпадений
* Вывод результатов в виде таблицы
* Скачивание результатов в формате Excel
* Сохранение и загрузка моделей

## Технологии
* Python 3.x
* pandas
* dask
* streamlit
* sentence-transformers
* torch

## Установка
**Клонирование репозитория:** 
git clone https://github.com/Kiraogi/Nero.git  
**Установить зависимости:** pip install -r requirements.txt  
**Запустить приложение:** streamlit run nero.py  

## Использование
* Загрузите Excel-файлы с данными
* Выберите колонки с названиями товаров
* Дообучите модель на примерах (если необходимо)
* Нажмите кнопку "Обработка данных"
* Результаты будут выведены в виде таблицы
* Скачайте результаты в формате Excel
