from abc import ABC, abstractmethod

class ProviderBase(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass