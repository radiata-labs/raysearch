from __future__ import annotations

from typing import Generic
from typing_extensions import TypeVar

from serpsage.models.app.base import BaseRequest, BaseResponse
from serpsage.models.base import MutableModel

RequestType = TypeVar("RequestType", bound=BaseRequest, default=BaseRequest)
ResponseType = TypeVar("ResponseType", bound=BaseResponse, default=BaseResponse)


class BaseStepContext(MutableModel, Generic[RequestType, ResponseType]):
    request_id: str = ""
    request: RequestType
    response: ResponseType
