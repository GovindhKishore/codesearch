import ollama
from codesearch.providers.base import BaseProvider

OLLAMA_MODEL = "llama3"

class OllamaProvider(BaseProvider):
    def __init__(self, model_name: str = OLLAMA_MODEL):
        self.model_name = model_name
        self.client = ollama.Client()

    def generate(self, prompt: str) -> str | None:
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": 0.1},
            )

            return response.response if response.response else None

        except Exception:
            return None