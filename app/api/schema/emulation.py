from pydantic import BaseModel


class EmulationUploadResponse(BaseModel):
    session_id: str
