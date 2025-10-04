# finam_hackathon
Команда Bandito-Gangsterito
<br>
## Инструкция по запуску

Предварительно склонируйте репозиторий
1. **Загрузка зависимостей**
```bash
pip install -r requirements.txt
```

2. **Настройка переменных окружения**
## Измените пути с датасетами в **.env** вручную или

```bash
# Путь до датасета с новостями
sed -i 's|NEWS_PATH=.*|NEWS_PATH=/your/new/path/to/news_data.csv|' .env

# Путь до датасета с временными рядами
sed -i 's|CANDLES_PATH=.*|CANDLES_PATH=/your/new/path/to/candles.csv|' .env

# Путь для output
sed -i 's|OUT_DIR=.*|OUT_DIR=/your/new/output/directory|' .env
```

3. **Запуск скрипта**
```python
python3 script.py
```
