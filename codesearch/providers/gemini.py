from google import genai
from google.genai import types
from codesearch.providers.base import BaseProvider

GEMINI_MODEL = "gemini-2.5-flash-lite"


class GeminiProvider(BaseProvider):
    def __init__(self, api_key: str, model_name: str = GEMINI_MODEL):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str) -> str | None:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1)
            )
            return response.text if response.text else None

        except Exception:
            return None
