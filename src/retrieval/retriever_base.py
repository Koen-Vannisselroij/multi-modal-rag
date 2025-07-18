from abc import ABC, abstractmethod
from .domain import TextRetrievalResult


class BaseTextRetriever(ABC):
    @abstractmethod
    def retrieve_texts(self, query, top_k=5) -> [TextRetrievalResult]:
        pass


class BaseImageRetriever(ABC):
    @abstractmethod
    def retrieve_images(self, query, top_k=5):
        pass


class BaseCollectionManager(ABC):
    @abstractmethod
    def add_documents(self, docs):
        pass

    @abstractmethod
    def get_collection(self):
        pass
