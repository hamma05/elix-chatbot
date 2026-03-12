# app/config.py
import os
ELIX_BASE_URL = "https://elix-demo.thinktank.de"
ELIX_WORKFLOW_SERVICE_PATH = ("/api-gateway/workflowcust-service/request/json-data")

REQUEST_TIMEOUT = 10

# MCP API Key (api key,default for testing)
MCP_API_KEY = os.getenv("MCP_API_KEY", "default-internal-key")

# Auth placeholders (not used yet)
ELIX_HEADERS = {
    "Content-Type": "application/json"
}


