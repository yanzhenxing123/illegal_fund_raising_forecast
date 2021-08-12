from pydantic import BaseModel, Field
from typing import Optional, Any


class Res(BaseModel):
    code: int = Field(...)
    msg: Optional[Any] = Field(...)
    data: Optional[Any] = Field(...)
