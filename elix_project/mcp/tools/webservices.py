# app/tools/webservices.py

import requests
from config import (
    ELIX_BASE_URL,
    ELIX_WORKFLOW_SERVICE_PATH,
    REQUEST_TIMEOUT,
    ELIX_HEADERS
)

 


def call_elix_webservice(webservice_id: int, method: str = "GET", body: dict | None = None, params: dict | None = None):
    """Generic Elix Web Service caller. Supports GET and POST"""
    url = f"{ELIX_BASE_URL}{ELIX_WORKFLOW_SERVICE_PATH}/{webservice_id}"

    try:
        response = requests.request(method=method.upper(),url=url,headers=ELIX_HEADERS,params=params,
                                    json=body,timeout=REQUEST_TIMEOUT)
        
        # Raise an exception for bad status codes
        response.raise_for_status()

        return {
            "webservice_id": webservice_id,
            "status_code": response.status_code,
            "data": response.json() if response.content else None  # ✅ Fixed: check if content exists
        }

    except requests.exceptions.RequestException as error:
        return {
            "error": "Elix webservice call failed",
            "details": str(error),
            "webservice_id": webservice_id
        }
    