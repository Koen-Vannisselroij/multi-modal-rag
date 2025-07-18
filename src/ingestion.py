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
                    docs.append({"filename": fname, "content": f.read()})
    return docs
