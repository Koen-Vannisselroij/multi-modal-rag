from abc import ABC

class MultiModalRetrieverException(ABC, Exception):
    code = "MMR_BASE"

class CollectionNotSetException(MultiModalRetrieverException):
    code = "MMR_COLLECTION_NOT_SET"

class CollectionAlreadyExistsException(MultiModalRetrieverException):
    def __init__(self, name, collection_type=None):
        self.name = name
        self.collection_type = collection_type
        super().__init__(f"Collection '{name}' already exists (type: {collection_type})")