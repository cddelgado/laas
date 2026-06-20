from __future__ import annotations

from fastapi import HTTPException


def openai_error(
    status_code: int,
    message: str,
    *,
    type_: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "error": {
                "message": message,
                "type": type_,
                "param": param,
                "code": code,
            }
        },
    )
