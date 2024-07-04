from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse
from app.api.v1.controllers.vexo_api_calls import get_serp_results, get_llm_response, get_llm_response_stream
from app.api.v1.models import  SerpRequest

vexo_api_router = APIRouter()

@vexo_api_router.post("/searchEvidence")
async def fetch_serp_results(
    request: SerpRequest = Body(...)
) -> StreamingResponse:
    return await get_serp_results(request)  

@vexo_api_router.post("/searchVexoo")
async def fetch_llm_response(
    request: SerpRequest = Body(...)
) -> StreamingResponse:
    return await get_llm_response(request)  

# @vexo_api_router.post("/searchVexooStream")
# async def fetch_llm_response(
#     request: SerpRequest = Body(...)
# ) -> StreamingResponse:
#     return await get_llm_response_stream(request)  


@vexo_api_router.post("/searchVexooStream")
async def fetch_llm_response(
    request: SerpRequest = Body(...)
) -> StreamingResponse:
    return await get_llm_response_stream(request)
