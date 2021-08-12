from pydantic import BaseModel, Field
from typing import Optional, Any, Dict


class Res(BaseModel):
    code: int = Field(...)
    msg: Optional[Any] = Field(...)
    data: Optional[Any] = Field(...)


def get_errors(errors: Dict):
    return errors.values()[0][0]

