""" Schemas for types used in this project. """

from typing import Tuple, Set, List, Dict, Optional, Union, Any, Iterable

from pydantic.fields import Field
from pydantic.main import BaseModel
from pydantic.types import StrictInt


class Entity(BaseModel):
    """Schema for single entity."""

    names: Set[str] = Field(..., title="All found aliases.")
    frequency: StrictInt = Field(0, title="Frequency in corpus.")
    description: Optional[str] = Field(None, title="Full description.")
    short_description: Optional[str] = Field(None, title="Short description.")
    quality: Optional[str] = Field(
        None, title="Level of quality of documents in which entity was found."
    )
    source_id: str = Field(..., title="ID of source document.")
    categories: Set[str] = Field(set(), title="Wiki categories.")
    pageviews: StrictInt = Field(0, title="Number of page views.")


class Annotation(BaseModel):
    """Schema for single annotation."""

    entity_name: str = Field(..., title="Entity name.")
    entity_id: Optional[str] = Field(None, title="Entity name.")
    start_pos: StrictInt = Field(..., title="Start character position.")
    end_pos: StrictInt = Field(..., title="End character position.")
