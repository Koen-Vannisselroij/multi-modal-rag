from .retriever_base import BaseTextRetriever, BaseImageRetriever


class ChromaMultiModalRetriever:
    def __init__(
        self, text_retriever: BaseTextRetriever, image_retriever: BaseImageRetriever
    ):
        self.text_retriever = text_retriever
        self.image_retriever = image_retriever

    def retrieve_texts(self, query, top_k=5):
        return self.text_retriever.retrieve_texts(query, top_k)

    def retrieve_images(self, query, top_k=5):
        return self.image_retriever.retrieve_images(query, top_k)
