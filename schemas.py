from dataclasses import dataclass, field
from typing import List, Optional, Dict

@dataclass
class SourceRef:
    filename: str
    page: int
    sheet: Optional[str] = None
    slide: Optional[str] = None
    bbox: Optional[List[float]] = None

@dataclass
class NormalizedBlock:
    block_id: str
    document_id: str
    type: str          # "text", "heading", "table", "image"
    text: str = ""
    table_data: Optional[Dict] = None
    source_ref: Optional[SourceRef] = None
    confidence: float = 1.0
    language: str = "en"
    metadata: Dict = field(default_factory=dict)