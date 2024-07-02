from fastapi import Request
from fastapi.responses import StreamingResponse
from app.api.v1.core import llm
from app.api.v1.web_crawler.search import call_search_engine
from app.api.v1.models import Response, SerpRequest, SerpAPIResponseBody
from app.api.v1.models import LLMResponse, LLMSummaryArgs, LLMSummaryRequest, LLMGeneratedResponse
from fastapi import HTTPException


async def get_serp_results(request: SerpRequest):
    args = request.input

    if args:
        question = args.query
        results = call_search_engine(question)
        response_body = SerpAPIResponseBody(response=results)
        return Response(success=True, response=response_body)
    return Response(success=False, response=SerpAPIResponseBody(response={}))


async def get_llm_response(request: LLMSummaryRequest):
    args = request.input

    if not args or not args.query:
        raise HTTPException(status_code=400, detail="Invalid input: query is required")

    try:
        links = call_search_engine(args.query)
        model_name = "azureai"
        check_premise = True
        answer, evidence_data, _ = llm.call_freshprompt(model_name, args.query, check_premise=True)
        evidence_list = []
        links_list = []
        links_and_evidences = {}

        print("------------------------------------")
        print(evidence_data)
        print("\n")
        print("Type of the Evidence data is:")
        print(type(evidence_data))
        print("------------------------------------")

        if 'organic_results' in links:
            organic_results = links['organic_results']
            for result in organic_results:
                snippet = result.get('snippet', '')
                link = result.get('link', '')
                if snippet and link:
                    evidence_list.append(snippet)
                    links_list.append(link)
                    links_and_evidences[link] = snippet

        response_body = LLMGeneratedResponse(
            answer=answer,
            evidences=evidence_data,
            links=links_list,
            links_and_evidences=links_and_evidences
        )

        return LLMResponse(success=True, response=response_body)

    except Exception as e:
        # Log the exception here
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


async def get_llm_response_stream(request: SerpRequest):
    args = request.input

    if not args or not args.query:
        raise HTTPException(status_code=400, detail="Invalid input: query is required")

    try:
        model_name = "azureai"
        
        async def generate():
            for chunk in llm.call_freshprompt_stream(model_name, args.query):
                yield chunk

        return StreamingResponse(generate(), media_type="text/plain")

    except Exception as e:
        # Log the exception here
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")