from .retriever_base import BaseTextRetriever, BaseImageRetriever
from .collection_manager import ChromaMultiModalCollectionManager
from .domain import CollectionType, TextRetrievalResult
from .exceptions import CollectionNotSetException
from logger import get_logger

logger = get_logger(__name__)


class ChromaTextRetriever(BaseTextRetriever):
    def __init__(self, collection_manager: ChromaMultiModalCollectionManager):
        self.collection_manager = collection_manager

    def retrieve_texts(
        self, query: str, top_k: int = 3, max_distance: float = 1.5
    ) -> list:
        collection = self.collection_manager._get_collection(CollectionType.TEXT)
        if collection is None:
            logger.warning("Text collection is not set.")
            raise CollectionNotSetException()
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = results.get("documents", [[]])[0]
        ids = results.get("ids", [[]])[0]
        metadatas = (
            results.get("metadatas", [[]])[0]
            if "metadatas" in results
            else [{} for _ in documents]
        )
        distances = (
            results.get("distances", [[]])[0]
            if "distances" in results
            else [None for _ in documents]
        )
        output = []
        for doc, doc_id, metadata, dist in zip(documents, ids, metadatas, distances):
            if dist is not None and dist < max_distance:
                output.append(
                    (
                        TextRetrievalResult(content=doc, id=doc_id, metadatas=metadata),
                        dist,
                    )
                )
        return output


class ChromaImageRetriever(BaseImageRetriever):
    def __init__(self, collection_manager: ChromaMultiModalCollectionManager):
        self.collection_manager = collection_manager

    def retrieve_images(self, query: str, top_k: int = 3, max_distance: float = 1.5):
        collection = self.collection_manager._get_collection(CollectionType.IMAGE)
        if collection is None:
            logger.warning("Image collection is not set.")
            raise CollectionNotSetException()
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = results.get("documents", [[]])[0]
        distances = (
            results.get("distances", [[]])[0]
            if "distances" in results
            else [None for _ in documents]
        )
        output = [
            (doc, dist)
            for doc, dist in zip(documents, distances)
            if dist is not None and dist < max_distance
        ]
        if not output:
            logger.info("No relevant images found for query: %s", query)
        return output
