"""Strict output alphabet policy; this does not assess semantic quality."""
FALLBACK = 'Я отвечаю только по-русски. Попробуй задать вопрос иначе.'


def russian_only(text):
    letters = [c.lower() for c in text if c.isalpha()]
    if not letters or any(not ('а' <= c <= 'я' or c == 'ё') for c in letters):
        return FALLBACK
    return text.strip()
