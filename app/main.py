import datetime
import logging
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from flask_cors import CORS

from app.api.api_router import api_router
from app.utils.models import RequestSizeLimitMiddleware, RootResponse

# Set up logging level
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(title="VEXOO API", version="0.0.1")

CORS(app)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
    logging.error(f"{request}: {exc_str}")
    content = {"status_code": 10422, "message": exc_str, "data": None}
    return JSONResponse(
        content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )


# Record server start time
server_start_time = datetime.datetime.now()


# Mount the API router, which handles versioning internally
app.include_router(api_router, prefix="/api")


@app.get("/", response_model=RootResponse)
def read_root():
    """
    API Root Endpoint.
    Returns a brief description of the API, the current timestamp, and the API's running time.
    """
    current_time = datetime.datetime.now()
    running_time = current_time - server_start_time
    running_time_str = str(running_time).split(".")[0]

    return {
        "message": "Welcome to the API. Use /docs for documentation.",
        "timestamp": current_time,
        "running_time": running_time_str,
    }

@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    """
    Custom endpoint to serve the OpenAPI schema.
    """
    return JSONResponse(content=app.openapi())
