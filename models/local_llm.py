# models/local_llm.py

import logging
from typing import List, Dict, Any
from llama_cpp import Llama

class ЛокальнаяНейроСеть:
    def __init__(self, путь_к_модели: str = "models/llm/mistral-7b-v0.1.Q4_K_M.gguf"):
        self.журнал = logging.getLogger('ЛокальнаяНейроСеть')
        try:
            self.модель = Llama(model_path=путь_к_модели, n_ctx=2048, n_threads=4)
            self.журнал.info(f"Мур-мур, модель Mistral 7B успешно загружена из {путь_к_модели}")
        except Exception as e:
            self.журнал.error(f"Мяу! Не удалось загрузить модель Mistral 7B: {e}")
            raise

    def генерировать(self, промпт: str, макс_токенов: int = 100) -> str:
        try:
            вывод = self.модель(промпт, max_tokens=макс_токенов, stop=["Человек:", "Котик:"])
            return вывод['choices'][0]['text'].strip()
        except Exception as e:
            self.журнал.error(f"Мяу-мяу, ошибка при генерации текста: {e}")
            return ""

    def получить_эмбеддинги(self, текст: str) -> List[float]:
        try:
            эмбеддинг = self.модель.embed(текст)
            return эмбеддинг.tolist()
        except Exception as e:
            self.журнал.error(f"Мур-мур, ошибка при создании эмбеддингов: {e}")
            return []

    def получить_инфо_модели(self) -> Dict[str, Any]:
        return {
            "имя_модели": "Mistral 7B",
            "путь_к_модели": self.модель.model_path,
            "размер_контекста": self.модель.n_ctx,
            "количество_потоков": self.модель.n_threads,
        }
