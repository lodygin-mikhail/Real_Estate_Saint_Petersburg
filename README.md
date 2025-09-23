# 🏠 Предсказание стоимости недвижимости Санкт-Петербурга / Real Estate Price Prediction in Saint-Petersburg

Проект для предсказания стоимости недвижимости в Санкт-Петербурге с использованием машинного обучения. 
Включает полный MLOps цикл: от сбора данных до мониторинга моделей и интерактивного веб-приложения.

## 📊 О проекте

Этот проект представляет собой end-to-end решение для предсказания цен на недвижимость в Санкт-Петербурге. 
Система включает:

- **Полный EDA и предобработку** данных с использованием DVC пайплайнов
- **Обучение нескольких ML-моделей** с трекингом экспериментов через MLflow
- **Версионирование моделей и данных** с помощью DVC и MinIO
- **REST API** на FastAPI для предсказаний
- **Интерактивный интерфейс** на Streamlit
- **Docker-контейнеризацию** для простого развертывания
- **Базу данных PostgreSQL** для хранения метаданных экспериментов
- **S3-совместимое хранилище** MinIO для артефактов

## 🚀 Быстрый старт

### Предварительные требования

- Docker
- Docker Compose
- Python 3.9+ (для локальной разработки)

### Запуск через Docker Compose

```bash
# Клонируйте репозиторий
git clone https://github.com/lodygin-mikhail/Real_Estate_Saint_Petersburg.git
cd Real_Estate_Saint_Petersburg

# Создайте файл .env на основе примера
cp .env.example .env

# Заполните переменные окружения в .env файле

# Запустите все сервисы
docker-compose up -d

# Или с выводом логов
docker-compose up
```

После запуска сервисы будут доступны:

- **FastAPI API:** http://localhost:6000
- **FastAPI Docs:** http://localhost:6000/docs
- **Streamlit App:** http://localhost:8501
- **MLflow UI:** http://localhost:5000
- **PgAdmin:** http://localhost:5050
- **MinIO Console:** http://localhost:9001

## 📁 Структура проекта

```text
Real_Estate_Saint_Petersburg/
├── .dvc                         # Служебная папка DVC
├── fastapi_service/             # FastAPI сервис
│   ├── fastapi_app.py           # Основное FastAPI приложение
│   ├── requirements.txt
│   └── Dockerfile
├── images/                      # Изображения для README.md
├── ml_pipeline/                 # ML пайплайны и DVC
│   ├── src/                     # Код подготовки данных и обучения
│   ├── data/                    # Данные (raw/interim/processed)
│   ├── models/                  # Обученные модели
│   ├── artifacts/               # Артефакты (уникальные значения и т.д.)
│   ├── notebooks/               # Jupyter ноутбуки
│   ├── reports/                 # Метрики модели
│   ├── dvc.yaml                 # DVC пайплайны
│   ├── dvc.lock                 # DVC метаданные
│   └── requirements.txt
├── mlflow_server/               # MLflow tracking server
│   └── Dockerfile
├── streamlit_service/           # Streamlit интерфейс
│   ├── img/                     # Изображения для интерфейса
│   ├── streamlit_app.py
│   ├── requirements.txt
│   └── Dockerfile
├── minio/                       # Данные MinIO
├── pgadmin/                     # Конфигурация PgAdmin
├── nginx.conf                   # Конфиг Nginx для MinIO
├── docker-compose.yml
├── .env.example                 # Шаблон переменных окружения
└── README.md
```

## 📸 Интерфейс Streamlit приложения
### 1. Форма ввода параметров
![screenshot](/images/streamlit_app.png)
_Интерфейс для ввода характеристик недвижимости_
### 2. Результат предсказания
![screenshot](/images/streamlit_app_2.png)
_Отображение предсказанной цены после нажатия кнопки "Предсказать цену"_

## 🛠️ Локальная разработка

### Настройка ML пайплайна
```bash
# Перейдите в папку ML пайплайна
cd ml_pipeline

# Установите зависимости
pip install -r requirements.txt
```

#### Настройки DVC remote
```bash
# Измените переменные в .dvc/config на свои
url = s3://your_bucket_name
endpointurl = https://s3.your_cloud_s3_storage
```
#### Установка переменных окружения для S3
```bash
# Для Linux:
export AWS_ACCESS_KEY_ID="your_aws_access_key_id" 
export AWS_SECRET_ACCESS_KEY="your_secret_access_key"
```

```bash
# Для Windows:
$env:AWS_ACCESS_KEY_ID="your_aws_access_key_id" 
$env:AWS_SECRET_ACCESS_KEY="your_secret_access_key"
```

```bash
# Загрузите данные
dvc pull

# Запустите пайплайн обработки данных
dvc repro
````

## 🐳 Docker команды
```bash
# Сборка и запуск всех сервисов
docker-compose up --build

# Сборка и запуск конкретного сервиса
docker-compose up --build fastapi-service
docker-compose up --build streamlit-web

# Остановка сервисов
docker-compose down

# Просмотр логов
docker-compose logs -f
docker-compose logs fastapi-service
docker-compose logs streamlit-web

# Перезапуск конкретного сервиса
docker-compose restart mlflow
```

## 📊 Управление ML экспериментами
### Работа с DVC пайплайнами

```bash
cd ml_pipeline

# Запуск полного пайплайна
dvc repro

# Запуск конкретной стадии
dvc repro src/models/prepare_dataset
dvc repro src/models/train

# Просмотр графа пайплайна
dvc dag
```
## 🔌 API Endpoints

### **Предсказание цены**
**POST** `/predict`

**Тело запроса:**
```json
{
    "flat_status": false,
    "num_of_rooms": 2,
    "total_area_m2": 45.5,
    "living_area_m2": 28.0,
    "kitchen_area_m2": 10.5,
    "floor": 5,
    "metro_station": "Площадь Восстания",
    "minutes_to_metro": 15,
    "transfer_type": "пешком",
    "house_age": 20,
    "is_future_building": false
}
```
**Ответ:**
```json
{
  "prediction": {
    "price": 7610252.757031128
  }
}
```
### Другие endpoints
- **GET** `/` - Информация о API
- **GET** `/health` - Проверка здоровья сервиса

## 📜 Лицензия
MIT License © 2025
