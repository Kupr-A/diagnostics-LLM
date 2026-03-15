
import os
import time
from openai import OpenAI

from src.config import LLM_MODEL_NAME


DEFAULT_SYSTEM_PROMPT = (
    "Ты медицинский AI-ассистент для исследовательской системы дифференциальной диагностики. "
    "Твоя задача — предложить наиболее вероятные диагностические гипотезы по жалобе пациента "
    "и похожим кейсам из базы. "
    "Используй похожие кейсы как ориентир, но не копируй их диагноз автоматически. "
    "Учитывай, что данные могут быть шумными и неполными. "
    "Не ставь окончательный диагноз. "
    "Отвечай строго одним JSON-объектом без markdown, без пояснений до и после JSON."
)


class OpenRouterClient:
    def __init__(self, api_key: str | None = None, model_name: str = LLM_MODEL_NAME):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "Переменная окружения OPENROUTER_API_KEY не установлена."
            )

        self.model_name = model_name
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.7,
        max_tokens: int = 1200,
        retries: int = 3,
        retry_delay: float = 2.0,
    ) -> str:
        last_error = None

        for attempt in range(1, retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                content = resp.choices[0].message.content
                if not content:
                    raise ValueError("OpenRouter вернул пустой ответ")

                return content.strip()

            except Exception as e:
                last_error = e
                if attempt < retries:
                    time.sleep(retry_delay)

        raise RuntimeError(f"Не удалось получить ответ от OpenRouter: {last_error}")