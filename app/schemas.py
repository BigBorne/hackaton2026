from pydantic import BaseModel
from typing import List

class AdInput(BaseModel):
    itemId: int
    mcId: int
    mcTitle: str
    description: str

class Draft(BaseModel):
    mcId: int
    mcTitle: str
    text: str

class ProcessingResponse(BaseModel):
    make_avert: bool
    drafts: List[Draft] = []