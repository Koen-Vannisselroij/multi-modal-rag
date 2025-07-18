import yaml
import sys
from ingestion import DataIngestor
from logger import get_logger
from main import get_retriever_and_manager

logger = get_logger(__name__)


def load_test_cases(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    # Expecting a list of dicts with 'question' and 'expected_filename' keys
    return [(item["question"], item["expected_filename"]) for item in data]


def setup_collections(collection_manager, docs, backend):
    if backend == "chroma":
        collection_manager.initiate_text_collection("default_text")
        collection_manager.add_text_knowledge(docs)
    elif backend == "tfidf":
        collection_manager.add_documents(docs)


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python evaluate_text_retrieval_accuracy.py <test_cases.yaml> <config.yaml>"
        )
        sys.exit(1)
    test_cases_path = sys.argv[1]
    config_path = sys.argv[2]
    test_cases = load_test_cases(test_cases_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    ingestor = DataIngestor(text_dir="data/evaluate")
    docs = ingestor.ingest_text_data()
    collection_manager, retriever = get_retriever_and_manager(config)
    setup_collections(collection_manager, docs, config["backend"])

    correct = 0
    print("\nRetrieval Accuracy Test Results:\n" + "-" * 40)
    for idx, (question, expected_filename) in enumerate(test_cases, 1):
        results = retriever.retrieve_texts(question, top_k=1)
        result_filename = results[0][0].filename if results else None

        if result_filename == expected_filename:
            status = "✓"
            correct += 1
        else:
            status = "✗"

        print(f"Q{idx}: {question}")
        print(
            f"   {status} Got: {result_filename or 'No result'} | Expected: {expected_filename}\n"
        )

    print("-" * 40)
    print(f"Retrieval accuracy: {correct}/{len(test_cases)} correct.\n")


if __name__ == "__main__":
    main()
