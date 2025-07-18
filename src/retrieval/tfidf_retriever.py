from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .retriever_base import BaseTextRetriever, BaseCollectionManager
from .domain import TextRetrievalResult
from logger import get_logger

logger = get_logger(__name__)

class TfidfCollectionManager(BaseCollectionManager):
    def __init__(self):
        self.docs = []
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = None

    def add_documents(self, docs):
        self.docs = docs
        texts = [doc["content"] for doc in docs]
        if texts:
            self.doc_vectors = self.vectorizer.fit_transform(texts)
        else:
            self.doc_vectors = None

    def get_collection(self):
        return self.docs, self.doc_vectors, self.vectorizer

class TfidfTextRetriever(BaseTextRetriever):
    def __init__(self, collection_manager: TfidfCollectionManager):
        self.collection_manager = collection_manager

    def retrieve_texts(self, query, top_k=1) -> [TextRetrievalResult]:
        docs, doc_vectors, vectorizer = self.collection_manager.get_collection()
        if not docs or doc_vectors is None:
            logger.warning("No documents available for retrieval.")
            return []
        query_vec = vectorizer.transform([query])
        sims = cosine_similarity(query_vec, doc_vectors).flatten()
        logger.info(f"Similarity scores: {sims}")
        top_indices = sims.argsort()[::-1][:top_k]
        results = []
        for i in top_indices:
            doc = docs[i]
            results.append(TextRetrievalResult(
                content=doc.get("content"),
                id=doc.get("id"),
                metadatas={"filename": doc.get("filename")} if doc.get("filename") else None,
            )
            )
        return results