from pydantic import BaseModel, ConfigDict

from serpsage.models.base import MutableModel


class BaseRequest(MutableModel):
    pass


class BaseResponse(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    request_id: str
