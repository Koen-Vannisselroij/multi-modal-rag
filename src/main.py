from ingestion import ingest_data
from retrieval import Retriever
from logger import get_logger

logger = get_logger(__name__)

def main():
    docs = ingest_data()
    if not docs:
        logger.error("No documents found in data/. Please add .txt files.")
        return
        
    retriever = Retriever()
    retriever.add_documents(docs)

    print("RAG pipeline ready. Type a query (or 'exit' to quit):")
    while True:
        query = input("Query: ").strip()
        if query.lower() in ("exit", "quit"): break
        results = retriever.retrieve(query)
        if results:
            print(f"Most relevant document: {results[0]['filename']}")
            print("---")
            print(results[0]['content'])
        else:
            print("No relevant documents found.")

if __name__ == "__main__":
    main() 