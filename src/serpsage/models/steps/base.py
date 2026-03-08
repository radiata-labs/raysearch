from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

from serpsage.models.base import MutableModel

if TYPE_CHECKING:
    from serpsage.models.app.base import BaseRequest, BaseResponse

RequestType = TypeVar("RequestType", bound="BaseRequest")
ResponseType = TypeVar("ResponseType", bound="BaseResponse")


class BaseStepContext(MutableModel, Generic[RequestType, ResponseType]):
    request_id: str = ""
    request: RequestType
    response: ResponseType
