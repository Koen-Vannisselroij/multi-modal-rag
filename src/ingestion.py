import os

class DataIngestor:
    def __init__(self, text_dir="data/sample_texts", image_dir="data/sample_images"):
        self.text_dir = text_dir
        self.image_dir = image_dir
        self.text_exts = (".txt",)
        self.image_exts = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")

    def ingest_files(self, data_dir, allowed_exts, process_file=None):
        docs = []
        for root, _, files in os.walk(data_dir):
            for fname in files:
                if fname.lower().endswith(allowed_exts):
                    file_path = os.path.join(root, fname)
                    content = process_file(file_path) if process_file else file_path
                    docs.append({"filename": fname, "content": content})
        return docs

    def ingest_text_data(self):
        def read_text_file(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        return self.ingest_files(self.text_dir, self.text_exts, process_file=read_text_file)

    def ingest_image_data(self):
        return self.ingest_files(self.image_dir, self.image_exts)