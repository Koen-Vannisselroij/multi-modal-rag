from pydantic import BaseModel
from typing import Optional, Dict, Any
from enum import Enum


class CollectionType(Enum):
    IMAGE = "image"
    TEXT = "text"


class CollectionItem(BaseModel):
    path: str
    media_type: CollectionType


class TextRetrievalResult(BaseModel):
    content: str
    id: Optional[str] = None
    metadatas: Optional[Dict[str, Any]] = None

    @property
    def filename(self) -> Optional[str]:
        return self.metadatas.get("filename") if self.metadatas else None
