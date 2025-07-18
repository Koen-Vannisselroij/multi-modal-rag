import sys
import yaml
from ingestion import DataIngestor
from retrieval import (
    TfidfCollectionManager,
    TfidfTextRetriever,
    ChromaMultiModalCollectionManager,
    ChromaMultiModalRetriever,
    ChromaImageRetriever,
    ChromaTextRetriever,
)
from logger import get_logger

logger = get_logger(__name__)

# --- Helper functions for dynamic instantiation ---


def get_chroma_client(client_type):
    import chromadb

    if client_type == "local":
        return chromadb.Client()
    elif client_type == "http":
        # You can add host/port to config.yaml if needed
        return chromadb.HttpClient(host="localhost", port=8000)
    else:
        raise ValueError(f"Unknown Chroma client type: {client_type}")


def get_embedding_function(embedding_type):
    if embedding_type == "sentence_transformers":

        class SentenceTransformerEmbeddingFunction:
            def __init__(self, model_name="all-MiniLM-L6-v2"):
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(model_name)

            def __call__(self, input):
                return self.model.encode(input).tolist()

        return SentenceTransformerEmbeddingFunction()
    elif embedding_type == "openai":

        class OpenAIEmbeddingFunction:
            def __call__(self, input):
                import openai

                return [
                    openai.embeddings.create(input=t, model="text-embedding-ada-002")
                    .data[0]
                    .embedding
                    for t in input
                ]

        return OpenAIEmbeddingFunction()
    else:
        raise ValueError(f"Unknown embedding function type: {embedding_type}")


def get_retriever_and_manager(config):
    backend = config["backend"]
    if backend == "chroma":
        client = get_chroma_client(config["chroma"]["client"])
        embedding_function = get_embedding_function(
            config["chroma"]["embedding_function"]
        )
        collection_manager = ChromaMultiModalCollectionManager(
            client, embedding_function
        )
        text_retriever = ChromaTextRetriever(collection_manager)
        image_retriever = ChromaImageRetriever(collection_manager)
        retriever = ChromaMultiModalRetriever(
            text_retriever=text_retriever, image_retriever=image_retriever
        )
    elif backend == "tfidf":
        collection_manager = TfidfCollectionManager()
        retriever = TfidfTextRetriever(collection_manager)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return collection_manager, retriever


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <config_path>")
        sys.exit(1)
    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    ingestor = DataIngestor()
    docs = ingestor.ingest_text_data()
    image_docs = ingestor.ingest_image_data()
    if not docs and not image_docs:
        logger.error(
            "No documents or images found in data/. Please add .txt files or images."
        )
        return

    collection_manager, retriever = get_retriever_and_manager(config)
    if config["backend"] == "chroma":
        if docs:
            collection_manager.initiate_text_collection("default_text")
            collection_manager.add_text_knowledge(docs)
        if image_docs:
            collection_manager.initiate_image_collection("default_image")
            collection_manager.add_image_knowledge(image_docs)
    elif config["backend"] == "tfidf":
        if docs:
            collection_manager.add_documents(docs)

    print("RAG pipeline ready. Type a query (or 'exit' to quit):")
    while True:
        query = input("Query: ").strip()
        if query.lower() in ("exit", "quit"):
            break

        # Text retrieval
        text_results = retriever.retrieve_texts(query)
        print("\n--- Text Retrieval Results ---")
        if text_results:
            for result, distance in text_results:
                print(f"{result} (distance: {distance:.4f})")
        else:
            print("No relevant documents found.")

        # Image retrieval
        image_results = retriever.retrieve_images(query)
        print("\n--- Image Retrieval Results ---")
        if image_results:
            for doc, distance in image_results:
                print(f"{doc} (distance: {distance:.4f})")
        else:
            print("No relevant images found.")
        print("\n" + "=" * 40 + "\n")


if __name__ == "__main__":
    main()
