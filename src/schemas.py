from pydantic import BaseModel, Field
from typing import Optional

class RoutingResult(BaseModel):
    route:str=Field(..., description="المسار المختار (Flash, Pro, Local Simulator, etc.)")
    response:str=Field(..., description="الرد النصي")
    money_saved: float = Field(default=0.0, description="المبلغ الذي تم توفيره")
    latency_ms:Optional[float]=Field(default=None, description="زمن الاستجابة بالمللي ثانية")