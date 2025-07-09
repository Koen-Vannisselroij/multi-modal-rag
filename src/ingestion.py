import os

def ingest_data(data_dir="data"):
    """
    Ingests text data from the data/ directory and all its subdirectories.
    Returns a list of document dicts: {'filename': ..., 'content': ...}
    """
    docs = []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith(".txt"):
                file_path = os.path.join(root, fname)
                with open(file_path, "r", encoding="utf-8") as f:
                    # Store relative path from data_dir for clarity
                    rel_path = os.path.relpath(file_path, data_dir)
                    docs.append({"filename": rel_path, "content": f.read()})
    return docs
