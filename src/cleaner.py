import re

from config import train_file


def clean_text(text):
    # Разделяем текст на строки по символу переноса строки
    lines = text.split('\n')

    cleaned_lines = []
    for line in lines:
        # Убираем все неалфавитные символы, кроме пробела, знаков препинания и букв
        cleaned_line = re.sub(r'[^А-Яа-яA-Za-z\s.,!?;:–-]', '', line)

        # Заменяем множественные пробелы на один
        cleaned_line = re.sub(r'\s+', ' ', cleaned_line)

        # Убираем пробелы в начале и в конце строки
        cleaned_line = cleaned_line.strip()

        # Добавляем очищенную строку в список
        cleaned_lines.append(cleaned_line)

    # Собираем текст обратно, разделяя его на строки
    cleaned_text = '\n'.join(cleaned_lines)

    return cleaned_text


# Открываем файл для чтения и записи
with open(train_file, 'r', encoding='utf-8') as file:
    # Чтение всего текста
    raw_text = file.read()

# Очистка текста
cleaned_text = clean_text(raw_text)

# Открываем файл для записи и сохраняем очищенный текст
with open(train_file, 'w', encoding='utf-8') as file:
    file.write(cleaned_text)

print("Файл успешно очищен и сохранен.")
