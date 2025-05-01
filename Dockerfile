FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt update && apt install -y python3 python3-pip git && \
    pip3 install --upgrade pip

# Копируем файлы проекта
WORKDIR /app
COPY . .

# Устанавливаем зависимости
RUN pip install -r requirements.txt

# Указываем точку входа
CMD ["python3", "src/train.py"]
