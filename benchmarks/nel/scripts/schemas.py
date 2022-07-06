""" Schemas for types used in this project. """

from typing import Set, Optional

from pydantic.fields import Field
from pydantic.main import BaseModel
from pydantic.types import StrictInt


class Entity(BaseModel):
    """Schema for single entity."""

    qid: str = Field(..., title="Wiki QID.")
    name: str = Field(..., title="Entity name.")
    aliases: Set[str] = Field(..., title="All found alias_entity_prior_probs.")
    count: StrictInt = Field(0, title="Count in Wiki corpus.")
    description: Optional[str] = Field(None, title="Full description.")
    article_title: Optional[str] = Field(None, title="Article title.")
    article_text: Optional[str] = Field(None, title="Article text.")


class Annotation(BaseModel):
    """Schema for single annotation."""

    entity_name: str = Field(..., title="Entity name.")
    entity_id: Optional[str] = Field(None, title="Entity ID.")
    start_pos: StrictInt = Field(..., title="Start character position.")
    end_pos: StrictInt = Field(..., title="End character position.")
