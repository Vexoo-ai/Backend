from fastapi import APIRouter
from app.api.v1.router import vexo_api_router

api_router = APIRouter()

api_router.include_router(vexo_api_router, prefix="/v1", tags=["VEXOO"])