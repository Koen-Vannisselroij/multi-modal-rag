import uuid
from typing import List
from chromadb.utils.data_loaders import ImageLoader
from .domain import CollectionItem, CollectionType
from .exceptions import CollectionAlreadyExistsException, CollectionNotSetException
from .retriever_base import BaseCollectionManager


class ChromaMultiModalCollectionManager(BaseCollectionManager):
    def __init__(self, client, embedding_function):
        self.client = client
        self.embedding_function = embedding_function
        self.collections = {CollectionType.IMAGE: None, CollectionType.TEXT: None}
        self.image_data_loader = None

    def _get_collection(self, collection_type: str):
        return self.collections[collection_type]

    def _set_collection(self, collection_type: str, collection):
        self.collections[collection_type] = collection

    def _initiate_collection(self, name: str, collection_type: str):
        collection = self._get_collection(collection_type)
        if collection is not None and collection.name == name:
            raise CollectionAlreadyExistsException(
                name, collection_type=collection_type
            )
        if collection_type == CollectionType.IMAGE:
            if self.image_data_loader is None:
                self.image_data_loader = ImageLoader
            collection = self.client.create_collection(
                name=name,
                embedding_function=self.embedding_function,
                data_loader=self.image_data_loader,
            )
        else:
            collection = self.client.create_collection(
                name=name,
                embedding_function=self.embedding_function,
            )
        self._set_collection(collection_type, collection)

    def initiate_image_collection(self, name: str):
        self._initiate_collection(name, CollectionType.IMAGE)

    def initiate_text_collection(self, name: str):
        self._initiate_collection(name, CollectionType.TEXT)

    def _add_knowledge(self, knowledge: List[CollectionItem], collection_type: str):
        collection = self._get_collection(collection_type)
        if collection is None:
            raise CollectionNotSetException()
        ids = [str(uuid.uuid4()) for _ in range(len(knowledge))]
        filenames = [item["filename"] for item in knowledge]
        print(filenames)
        collection.add(
            ids=ids,
            documents=[doc["content"] for doc in knowledge],
            metadatas=[{"filename": filename} for filename in filenames],
        )

    def add_image_knowledge(self, knowledge: List[CollectionItem]):
        self._add_knowledge(knowledge, CollectionType.IMAGE)

    def add_text_knowledge(self, knowledge: List[CollectionItem]):
        self._add_knowledge(knowledge, CollectionType.TEXT)

    # For interface compatibility with BaseCollectionManager
    def add_documents(self, docs):
        self.add_text_knowledge(docs)

    def get_collection(self):
        # Return the text collection for compatibility
        return self._get_collection(CollectionType.TEXT), None, None
