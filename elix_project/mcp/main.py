# app/main.py
from fastapi import FastAPI, Header, HTTPException, Body, Request
from tools import call_elix_webservice

from config import MCP_API_KEY

app = FastAPI(
    title="Elix MCP",
    description="MCP (Model Context Protocol) server for the Elix progress tracking system",
    version="1.0.0"
)

# ---- Authentication ----
def check_auth(x_api_key: str):
    if x_api_key != MCP_API_KEY:
        raise HTTPException(status_code=401, detail="Wrong API Key")
    

# ---- Health Check ----
@app.get("/")
def health_check():
    return {"status": "Allahou Akbar ,Its Working!"}

#function to detetct method type

def detect_method(service_id: int):
    # Example using local mapping
    methods = {
        1161020185: "POST",
        14: "GET",
    }
    return methods.get(service_id, "GET")

# ---- Web Services Tool ----
@app.api_route("/tools/webservices/{webservice_id}",methods=["GET", "POST"])
async def elix_webservice(webservice_id: int,request: Request,x_api_key: str = Header(None)):
    """Single endpoint for calling Elix web services. Supports GET (query params) and POST (JSON body)."""
    check_auth(x_api_key) # Authentication check

    # Determine HTTP method used by the client
    method = detect_method(webservice_id)
    # Extract query parameters for GET
    params = dict(request.query_params) if (method == "GET") else None

    # Extract JSON body for POST
    body = await request.json() if (method == "POST") else None

    # Call the generic webservice tool
    return call_elix_webservice(webservice_id=webservice_id,method=method,body=body,params=params)
