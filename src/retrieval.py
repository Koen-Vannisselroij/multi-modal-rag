import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from logger import get_logger

logger = get_logger(__name__)

class Retriever:
    def __init__(self):
        self.docs = []
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = None

    def add_documents(self, docs):
        self.docs = docs
        texts = [doc['content'] for doc in docs]
        if texts:
            self.doc_vectors = self.vectorizer.fit_transform(texts)
        else:
            self.doc_vectors = None

    def retrieve(self, query, top_k=1):
        if not self.docs or self.doc_vectors is None:
            logger.warning("No documents available for retrieval.")
            return []
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.doc_vectors).flatten()
        logger.info(f"Similarity scores: {sims}")
        top_indices = sims.argsort()[::-1][:top_k]
        return [self.docs[i] for i in top_indices]