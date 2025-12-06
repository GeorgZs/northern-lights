"""
LangChain agent service for company queries with database search and validation.

Main flow:
1. Agentically classify query (using Gemini) - simple lookup vs complex query
2. If simple → Search database and handle directly
3. If complex → Forward to Neo4j agent
4. If not found → Trigger ingestion (with web lookup for company names)
"""

import asyncio

import logging
import os
import json
import re
from concurrent.futures import ThreadPoolExecutor

import httpx

from app.db.queries import company_queries
from app.services.portfolio_ingestion import ingest_company_with_portfolio

logger = logging.getLogger(__name__)

NEO_AGENT_URL = os.getenv("NEO_AGENT_INVOKE")
AURA_CLIENT_ID = os.getenv("AURA_CLIENT_ID")
AURA_CLIENT_SECRET = os.getenv("AURA_CLIENT_SECRET")


def extract_final_text(log):
    """
    Extract the final user-facing text from an agent event log.
    Supports logs that include:
    - {"text": "..."}
    - {"message": "..."}
    - {"output": {"text": "..."}}

    Returns: str or None
    """
    if not isinstance(log, list):
        return None

    def get_text(entry):
        if not isinstance(entry, dict):
            return None

        # PRIORITY 1: Direct text response (the final answer)
        # This is the final user-facing message from the agent
        if entry.get("type") == "text" and entry.get("text"):
            return entry["text"]

        # PRIORITY 2: Common alt key
        if entry.get("message") and not entry.get("type"):
            return entry["message"]

        # PRIORITY 3: Sometimes tool outputs embed text
        # Only use this if no type field (avoid matching tool results)
        if not entry.get("type"):
            output = entry.get("output")
            if isinstance(output, dict) and isinstance(output.get("text"), str):
                return output["text"]

        return None

    # Search from the end backwards
    for entry in reversed(log):
        text = get_text(entry)
        if text:
            return text.strip()

    return None


# Try to import Gemini for agentic query classification
try:
    import google.generativeai as genai

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        try:
            gemini_model = genai.GenerativeModel("gemini-2.0-flash-exp")
        except:
            gemini_model = genai.GenerativeModel("gemini-1.5-pro")
    else:
        gemini_model = None
        logger.warning("GEMINI_API_KEY not set - query classification will fall back to pattern matching")
except ImportError:
    gemini_model = None
    logger.warning("Gemini not available - query classification will fall back to pattern matching")


class InputValidator:
    """Agentically validates user input and determines query type using Gemini"""

    # Organization ID pattern: UUID format OR Swedish org number (10 digits with optional dash)
    UUID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
    SWEDISH_ORG_PATTERN = re.compile(r"^\d{6}-?\d{4}$")

    @classmethod
    def _classify_query_agentic(cls, query: str) -> dict[str, any]:
        """
        Use Gemini to agentically classify the query type.
        Handles misspellings and understands intent.
        """
        if not gemini_model:
            # Fallback to pattern matching if Gemini not available
            return cls._classify_query_fallback(query)

        try:
            prompt = f"""You are a query classifier for a Swedish company and fund database system.

Analyze this user query and classify it into one of these types:

1. "org_id" - The query is clearly a Swedish organization number (10 digits, format XXXXXX-XXXX) or UUID
2. "company_name" - The query is a simple company name lookup (1-5 words, just asking to find a company)
3. "general_query" - The query is a complex question that requires graph traversal, relationship analysis, or multi-step reasoning (e.g., "Who owns X?", "What else does Y own?", "Show me all funds in tech sector")

Examples:
- "556043-4200" → org_id
- "Investor AB" → company_name
- "ericsson" → company_name (simple lookup)
- "Who owns ericsson?" → general_query (complex question)
- "Who owns ericsson and what else do they also own?" → general_query (multi-part question)
- "Show me all companies owned by Investor AB" → general_query (graph query)
- "What funds invest in tech startups?" → general_query (complex filter/analysis)

IMPORTANT:
- Handle misspellings intelligently (e.g., "ericsson" vs "eriksson" - both are company_name)
- Don't rely on question marks - understand intent
- Simple lookups (just a company name) should be company_name
- Complex questions requiring graph analysis should be general_query

Query: "{query}"

Return JSON only with this structure:
{{
    "type": "org_id" | "company_name" | "general_query",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}
"""

            response = gemini_model.generate_content(
                prompt, generation_config={"response_mime_type": "application/json"}
            )

            response_text = response.text.strip()
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1]) if len(lines) > 2 else response_text

            classification = json.loads(response_text)
            query_type = classification.get("type", "general_query")
            confidence = classification.get("confidence", 0.5)
            reasoning = classification.get("reasoning", "")

            logger.debug(
                f"Agentic classification: type={query_type}, confidence={confidence:.2f}, reasoning={reasoning}"
            )

            return {"type": query_type, "confidence": confidence, "reasoning": reasoning}

        except Exception as e:
            logger.warning(f"Agentic classification failed: {e}, falling back to pattern matching")
            return cls._classify_query_fallback(query)

    @classmethod
    def _classify_query_fallback(cls, query: str) -> dict[str, any]:
        """Fallback pattern-based classification if Gemini fails"""
        query = query.strip()

        # Check if it's a UUID
        if cls.UUID_PATTERN.match(query):
            return {"type": "org_id", "confidence": 1.0, "reasoning": "Matches UUID pattern"}

        # Check if it's a Swedish organization number
        cleaned_org_id = re.sub(r"[-\s]", "", query)
        if cls.SWEDISH_ORG_PATTERN.match(query) or (len(cleaned_org_id) == 10 and cleaned_org_id.isdigit()):
            formatted = f"{cleaned_org_id[:6]}-{cleaned_org_id[6:]}"
            return {"type": "org_id", "confidence": 1.0, "reasoning": "Matches Swedish org number pattern"}

        # Check if it's a complex query (contains question words or multiple clauses)
        question_words = ["who", "what", "where", "when", "why", "how", "which", "show", "list", "find", "tell"]
        has_question = any(word in query.lower() for word in question_words)
        has_multiple_clauses = len(query.split()) > 5 or (" and " in query.lower() or " or " in query.lower())
        
        if has_question or has_multiple_clauses:
            return {"type": "general_query", "confidence": 0.8, "reasoning": "Contains question words or multiple clauses"}

        # Check if it's a simple company name (1-5 words, alphanumeric with common company suffixes)
        words = query.split()
        if 1 <= len(words) <= 5:
            # Simple pattern: alphanumeric, spaces, common company suffixes
            company_pattern = re.compile(r"^[A-Za-zÅÄÖåäö0-9\s&\-\.]{1,100}$")
            if company_pattern.match(query):
                return {"type": "company_name", "confidence": 0.9, "reasoning": "Simple company name pattern (1-5 words)"}

        # Default to general_query for fallback (safer to forward to Neo4j agent)
        return {"type": "general_query", "confidence": 0.3, "reasoning": "Fallback: defaulting to general query"}

    @classmethod
    def validate_input(cls, query: str) -> dict[str, any]:
        """
        Agentically validate user input and determine query type.

        Types:
        - "org_id": UUID or Swedish organization number
        - "company_name": Simple company name lookup
        - "general_query": Complex question requiring Neo4j agent

        Returns:
            dict with keys: valid (bool), type (str), cleaned (str), error (str)
        """
        query = query.strip()

        if not query:
            return {"valid": False, "type": None, "cleaned": None, "error": "Empty query"}

        # Use agentic classification
        classification = cls._classify_query_agentic(query)
        query_type = classification["type"]

        # Clean and format based on type
        if query_type == "org_id":
            # Format Swedish org numbers consistently
            cleaned_org_id = re.sub(r"[-\s]", "", query)
            if len(cleaned_org_id) == 10 and cleaned_org_id.isdigit():
                cleaned = f"{cleaned_org_id[:6]}-{cleaned_org_id[6:]}"
            else:
                cleaned = query.lower()  # UUID
            return {"valid": True, "type": "org_id", "cleaned": cleaned, "error": None}

        elif query_type == "company_name":
            # Keep original query for company name (handles misspellings)
            return {"valid": True, "type": "company_name", "cleaned": query, "error": None}

        else:  # general_query
            # Keep original query for forwarding to Neo4j agent
            return {"valid": True, "type": "general_query", "cleaned": query, "error": None}


class CompanyAgentTools:
    """Tools for the LangChain agent"""

    def __init__(self):
        self.neo_agent_url = NEO_AGENT_URL

    def _convert_neo4j_to_json(self, obj):
        """Convert Neo4j objects to JSON-serializable format"""
        from neo4j.time import DateTime

        if isinstance(obj, DateTime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._convert_neo4j_to_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_neo4j_to_json(item) for item in obj]
        else:
            return obj

    async def search_database(self, query: str) -> str:
        """
        Search database for company by name or organization ID.

        Args:
            query: Company name or organization ID

        Returns:
            JSON string with search results
        """
        try:
            # Try by organization ID first
            company = company_queries.get_company(query)
            if company:
                company_json = self._convert_neo4j_to_json(company)
                return json.dumps({"found": True, "data": company_json, "query_type": "org_id"})

            # Try by company name using the proper function
            company = company_queries.find_company_by_name(query)
            if company:
                logger.debug(f"Company found: type={type(company)}, keys={list(company.keys()) if isinstance(company, dict) else 'not a dict'}")
                logger.debug(f"Company name: {repr(company.get('name'))}, company_id: {repr(company.get('company_id'))}")
                company_json = self._convert_neo4j_to_json(company)
                logger.debug(f"After conversion - name: {repr(company_json.get('name'))}, company_id: {repr(company_json.get('company_id'))}")
                return json.dumps({"found": True, "data": company_json, "query_type": "company_name"})
            else:
                return json.dumps({"found": False, "query": query, "query_type": "company_name"})

        except Exception as e:
            logger.error(f"Error searching database: {e}")
            return json.dumps({"found": False, "error": f"Database search error: {str(e)}"})

    async def _get_neo4j_token(self) -> str | None:
        """
        Fetch Neo4j OAuth access token using client credentials.

        Returns:
            Access token or None if credentials not configured
        """
        if not AURA_CLIENT_ID or not AURA_CLIENT_SECRET:
            logger.warning("AURA_CLIENT_ID or AURA_CLIENT_SECRET not configured")
            return None

        # According to Neo4j Aura API docs: https://neo4j.com/docs/aura/classic/platform/api/authentication/
        # The token endpoint requires HTTP Basic Authentication (not credentials in body)
        # Client ID = username, Client Secret = password
        try:
            token_url = "https://api.neo4j.io/oauth/token"

            # HTTP Basic Auth: base64 encode client_id:client_secret
            import base64

            credentials = base64.b64encode(f"{AURA_CLIENT_ID}:{AURA_CLIENT_SECRET}".encode()).decode()

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    token_url,
                    headers={
                        "Authorization": f"Basic {credentials}",
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                    data={"grant_type": "client_credentials"},
                )
                response.raise_for_status()
                token_data = response.json()
                access_token = token_data.get("access_token")
                if access_token:
                    logger.info("Successfully obtained Aura API access token")
                    return access_token
                else:
                    logger.error(f"Token response missing access_token: {token_data}")
                    return None

        except httpx.HTTPStatusError as e:
            logger.error(f"Aura API token endpoint returned {e.response.status_code}: {e.response.text[:200]}")
            if e.response.status_code == 401:
                logger.error("Invalid credentials - check AURA_CLIENT_ID and AURA_CLIENT_SECRET")
            elif e.response.status_code == 403:
                logger.error("Credentials may not have permission to access the API")
            return None
        except Exception as e:
            logger.error(f"Error obtaining Aura API token: {e}", exc_info=True)
            return None

    async def query_neo4j_agent(self, company_data: str) -> str:
        """
        Query Neo4j agent for detailed company information.

        Args:
            company_data: JSON string with company data from database

        Returns:
            JSON string with Neo4j agent response
        """

        if not self.neo_agent_url:
            return json.dumps({"error": "NEO_AGENT_INVOKE not configured"})

        # Fetch OAuth token
        token = await self._get_neo4j_token()
        if not token:
            error_msg = "Failed to obtain Neo4j OAuth token. Check AURA_CLIENT_ID and AURA_CLIENT_SECRET."
            return json.dumps({"error": error_msg})

        try:
            data = json.loads(company_data)
            company_id = data.get("company_id") or data.get("organization_id")

            # Prepare request with Bearer token
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

            payload = {"input": f"Find information about company with ID: {company_id}"}

            logger.debug(f"Company query payload: {payload}")
            logger.debug(f"Company query headers: {dict(headers)}")

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.neo_agent_url, json=payload, headers=headers)
                logger.debug(f"Company query response status: {response.status_code}")
                if response.status_code != 200:
                    logger.debug(f"Company query response body: {response.text[:500]}")
                response.raise_for_status()
                return json.dumps(response.json())

        except httpx.HTTPStatusError as e:
            logger.error(f"Neo4j agent HTTP error: {e}")
            return json.dumps({"error": f"Neo4j agent authentication failed: {e.response.status_code}"})
        except Exception as e:
            logger.error(f"Error querying Neo4j agent: {e}")
            return json.dumps({"error": f"Neo4j agent query failed: {str(e)}"})

    async def query_neo4j_agent_general(self, query: str) -> str:
        """
        Query Neo4j agent with a general question/query.
        Extracts only the final text response from the agent log.

        Args:
            query: Natural language question or query

        Returns:
            JSON string with extracted text response or error
        """
        import json

        logger.info(f"Querying Neo4j agent with general query: {query}")

        if not self.neo_agent_url:
            return json.dumps({"error": "NEO_AGENT_INVOKE not configured"})

        token = await self._get_neo4j_token()
        if not token:
            error_msg = "Failed to obtain Neo4j OAuth token. Check AURA_CLIENT_ID and AURA_CLIENT_SECRET."
            return json.dumps({"error": error_msg})

        try:
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            payload = {"input": query}

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(self.neo_agent_url, json=payload, headers=headers)
                response.raise_for_status()

                # Get the raw response
                raw_result = response.json()
                logger.debug(f"Raw Neo4j agent response type: {type(raw_result)}")
                logger.debug(
                    f"Raw Neo4j agent response keys: {raw_result.keys() if isinstance(raw_result, dict) else 'not a dict'}"
                )

                # Handle wrapper format with 'content' field
                if isinstance(raw_result, dict) and "content" in raw_result:
                    content = raw_result["content"]
                    logger.debug(f"Found 'content' field, type: {type(content)}")

                    # Content should be the actual log array
                    if isinstance(content, list):
                        final_text = extract_final_text(content)
                        if final_text:
                            logger.info(f"✓ Extracted final text from agent log")
                            return json.dumps({"text": final_text})
                        else:
                            logger.error(f"❌ Failed to extract text from {len(content)} log items")
                            logger.debug(f"Log items: {content}")
                            return json.dumps({"error": "No text response found in agent log"})

                    # Content might be a string already
                    elif isinstance(content, str):
                        logger.info(f"✓ Content is already a string")
                        return json.dumps({"text": content})

                    else:
                        logger.error(f"Unexpected content type: {type(content)}")
                        return json.dumps({"error": f"Unexpected content type: {type(content)}"})

                # Handle direct list format (original behavior)
                elif isinstance(raw_result, list):
                    final_text = extract_final_text(raw_result)
                    if final_text:
                        logger.info(f"✓ Extracted final text from agent log")
                        return json.dumps({"text": final_text})
                    else:
                        logger.error(f"❌ Failed to extract text from {len(raw_result)} log items")
                        return json.dumps({"error": "No text response found in agent log"})

                # Handle direct dict with text field
                elif isinstance(raw_result, dict) and raw_result.get("text"):
                    logger.info(f"✓ Found direct text field")
                    return json.dumps({"text": raw_result["text"]})

                else:
                    logger.error(
                        f"Unexpected response format. Keys: {list(raw_result.keys()) if isinstance(raw_result, dict) else 'not a dict'}"
                    )
                    return json.dumps({"error": "Invalid response format from agent"})

        except httpx.TimeoutException:
            logger.error(f"Neo4j agent request timed out after 60 seconds")
            return json.dumps({"error": "Query timed out. The Neo4j agent took too long to respond."})
        except httpx.HTTPStatusError as e:
            logger.error(f"Neo4j agent HTTP error: {e.response.status_code} - {e.response.text}")
            return json.dumps(
                {"error": f"Neo4j agent returned error {e.response.status_code}: {e.response.text[:200]}"}
            )
        except httpx.RequestError as e:
            logger.error(f"Neo4j agent request error: {e}")
            return json.dumps({"error": f"Failed to connect to Neo4j agent: {str(e)}"})
        except Exception as e:
            logger.error(f"Error querying Neo4j agent: {e}", exc_info=True)
            return json.dumps({"error": f"Neo4j agent query failed: {str(e)}"})

    async def trigger_ingestion(self, query: str) -> str:
        """
        Trigger ingestion pipeline for a company.

        If company name provided, uses web lookup to find organization number.

        Args:
            query: Organization ID or company name

        Returns:
            JSON string with ingestion results
        """
        # Validate input
        validation = InputValidator.validate_input(query)
        if not validation["valid"]:
            return f'{{"status": "error", "error": "{validation["error"]}"}}'

        query_type = validation["type"]
        cleaned_query = validation["cleaned"]

        organization_id = None

        # If company name, look up organization number from web
        if query_type == "company_name":
            logger.info(f"Looking up organization number for company name: {cleaned_query}")

            try:
                from app.services.portfolio_ingestion import (
                    gemini_model,
                    lookup_org_number_from_web,
                    tavily_client,
                )

                # Check if Tavily and Gemini are available
                if not tavily_client or not gemini_model:
                    logger.warning("Tavily or Gemini not available - cannot lookup org number")
                    return json.dumps(
                        {
                            "status": "error",
                            "error": (
                                f"Company '{cleaned_query}' not found -- "
                                "Agent ingest is not possible, try again later. (Tavily/Gemini API not configured)"
                            ),
                        }
                    )

                # Use web lookup to find org number
                organization_id = lookup_org_number_from_web(cleaned_query)

                if not organization_id:
                    error_msg = (
                        f"Could not find organization number for company '{cleaned_query}'. "
                        "Please provide the Swedish organization number (format: XXXXXX-XXXX) manually."
                    )
                    return f'{{"status": "error", "error": "{error_msg}"}}'

                logger.info(f"Found organization number {organization_id} for {cleaned_query}")

            except Exception as e:
                logger.error(f"Error looking up organization number: {e}")
                return json.dumps(
                    {
                        "status": "error",
                        "error": (
                            f"Company '{cleaned_query}' not found -- "
                            f"Agent ingest is not possible, try again later. (Error: {str(e)})"
                        ),
                    }
                )
        else:
            # Already have org ID
            organization_id = cleaned_query

        try:
            # Log progress
            logger.info(f"Starting ingestion for {organization_id} ({cleaned_query})")

            # Run ingestion in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(
                    executor, ingest_company_with_portfolio, organization_id, cleaned_query
                )

            logger.info(
                f"Ingestion completed for {organization_id}: {len(result['portfolio'])} portfolio companies found"
            )

            return json.dumps(
                {
                    "status": "completed",
                    "organization_id": result["organization_id"],
                    "portfolio_companies_found": len(result["portfolio"]),
                    "companies_processed": result["companies_processed"],
                }
            )

        except Exception as e:
            logger.error(f"Error during ingestion for {organization_id}: {e}")

            return json.dumps({"status": "error", "error": str(e), "organization_id": organization_id})


class AgentResponse:
    """Response from the agent"""

    def __init__(
        self,
        message: str,
        company_found: bool = False,
        company_data: dict[str, any] | None = None,
        ingestion_triggered: bool = False,
        ingestion_result: dict[str, any] | None = None,
        error: str | None = None,
    ):
        # Ensure message is always a string
        if not isinstance(message, str):
            if isinstance(message, list):
                # If it's a list, try to extract text from it

                for item in reversed(message):
                    if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                        message = str(item["text"])
                        break
                else:
                    message = json.dumps(message)  # Fallback: stringify the list
            else:
                message = str(message)

        self.message = message
        self.company_found = company_found
        self.company_data = company_data
        self.ingestion_triggered = ingestion_triggered
        self.ingestion_result = ingestion_result
        self.error = error

    def to_dict(self) -> dict[str, any]:
        return {
            "message": self.message,
            "company_found": self.company_found,
            "company_data": self.company_data,
            "ingestion_triggered": self.ingestion_triggered,
            "ingestion_result": self.ingestion_result,
            "error": self.error,
        }


async def process_query(query: str) -> AgentResponse:
    """
    Process user query using LangChain agent.

    Flow:
    1. Validate input and determine type
    2. If general_query → Query Neo4j agent directly
    3. If company_name/org_id → Search database
    4. If found → Query Neo4j agent
    5. If not found → Trigger ingestion (if org ID provided)

    Args:
        query: User query (company name, organization ID, or general question)

    Returns:
        AgentResponse with results
    """
    import json  # ← Move import to the TOP of the function

    logger.info(f"Processing query: {query}")

    # Step 1: Validate input
    validation = InputValidator.validate_input(query)
    if not validation["valid"]:
        return AgentResponse(message=validation["error"], company_found=False, error=validation["error"])

    query_type = validation["type"]
    cleaned_query = validation["cleaned"]

    tools = CompanyAgentTools()

    try:
        # Step 2: If it's a general query, send directly to Neo4j agent
        if query_type == "general_query":
            logger.info(f"Processing as general query: {cleaned_query}")

            # Query Neo4j agent - now returns simplified {"text": "..."} format
            neo4j_result_str = await tools.query_neo4j_agent_general(cleaned_query)

            try:
                neo4j_result = json.loads(neo4j_result_str)

                # Check for error
                if neo4j_result.get("error"):
                    return AgentResponse(
                        message=f"Error: {neo4j_result['error']}",
                        company_found=False,
                        error=neo4j_result["error"],
                    )

                # Extract text from simplified response
                message = neo4j_result.get("text", "No response text found.")
                logger.info(f"✓ Received text response (length: {len(message)})")

                return AgentResponse(
                    message=message,
                    company_found=False,
                    company_data=None,
                )

            except json.JSONDecodeError:
                # If not JSON, return as plain text
                return AgentResponse(
                    message=str(neo4j_result_str),
                    company_found=False,
                    company_data=None,
                )

        # Step 3: For company_name/org_id, search database
        ## if we do not pass to agent we just no id or name lookup
        search_result_str = await tools.search_database(query)
        search_result = json.loads(search_result_str)

        logger.info(f"Search result: {search_result}")

        # Step 4: If found, validate company data and query Neo4j agent
        if search_result.get("found"):
            company_data = search_result.get("data")
            logger.debug(f"Company search returned data with keys: {list(company_data.keys()) if company_data else 'None'}")
            logger.debug(f"Company data values - name: {repr(company_data.get('name'))}, company_id: {repr(company_data.get('company_id'))}, organization_id: {repr(company_data.get('organization_id'))}")
            
            company_name = company_data.get("name")
            company_id = company_data.get("company_id") or company_data.get("organization_id")
            
            # Validate that we have essential company data
            # company_id is required (must exist on the node)
            # company_name can be None/empty for some nodes, but we'll use a fallback
            if not company_id or company_id is None:
                logger.warning(
                    f"Company search returned data but missing company_id for '{cleaned_query}': "
                    f"name={repr(company_name)}, id={repr(company_id)}. Treating as not found."
                )
                # Fall through to ingestion step
            else:
                # Use company_id as fallback name if name is missing
                if not company_name or company_name is None:
                    company_name = company_id
                    logger.debug(f"Using company_id as name fallback: {company_name}")
                # Build detailed company info in markdown
                company_details = f"## ✓ Company Found: **{company_name}**\n\n"
                company_details += f"**Organization ID:** `{company_id}`\n\n"

                if company_data.get("description"):
                    company_details += f"**Description:**\n{company_data['description']}\n\n"

                if company_data.get("sectors") and len(company_data["sectors"]) > 0:
                    sectors = company_data["sectors"]
                    if isinstance(sectors, list):
                        # Filter out None values before joining
                        valid_sectors = [str(s) for s in sectors if s is not None]
                        if valid_sectors:
                            company_details += f"**Sectors:** {', '.join(valid_sectors)}\n\n"
                    else:
                        if sectors is not None:
                            company_details += f"**Sectors:** {sectors}\n\n"

                details_added = False
                if company_data.get("website"):
                    company_details += f"**Website:** [{company_data['website']}]({company_data['website']})\n"
                    details_added = True
                if company_data.get("year_founded"):
                    company_details += f"**Founded:** {company_data['year_founded']}\n"
                    details_added = True
                if company_data.get("num_employees"):
                    company_details += f"**Employees:** {company_data['num_employees']}\n"
                    details_added = True
                if company_data.get("country_code"):
                    company_details += f"**Country:** {company_data['country_code']}\n"
                    details_added = True

                if details_added:
                    company_details += "\n"

                # Query Neo4j agent for additional information
                logger.info(f"Querying Neo4j agent for additional info about {company_name} ({company_id})")
                neo4j_result_str = await tools.query_neo4j_agent(json.dumps(company_data))

                try:
                    neo4j_result = json.loads(neo4j_result_str)
                except json.JSONDecodeError as e:
                    logger.warning(f"Neo4j agent response is not JSON: {e}, using database data only")
                    company_details += "*Showing information from our database.*"
                    return AgentResponse(
                        message=company_details,
                        company_found=True,
                        company_data=company_data,
                    )

                if neo4j_result.get("error"):
                    logger.warning(f"Neo4j agent returned error: {neo4j_result.get('error')}, using database data only")
                    company_details += "*Showing information from our database.*"
                    return AgentResponse(
                        message=company_details,
                        company_found=True,
                        company_data=company_data,
                    )

                logger.info(f"Neo4j agent provided additional insights for {company_name}")
                company_details += "*Additional insights from our knowledge graph...*"
                return AgentResponse(
                    message=company_details,
                    company_found=True,
                    company_data=neo4j_result.get("data", company_data),
                )

        # Step 5: Not found - trigger ingestion
        logger.info(f"Company not found with query '{cleaned_query}' - triggering ingestion")
        ingestion_result_str = await tools.trigger_ingestion(cleaned_query)
        ingestion_result = json.loads(ingestion_result_str)

        if ingestion_result.get("status") == "completed":
            portfolio_count = ingestion_result.get("portfolio_companies_found", 0)
            org_id = ingestion_result.get("organization_id")
            companies_processed = ingestion_result.get("companies_processed", 0)

            if query_type == "company_name":
                message = (
                    f"## Company Not Found: *{cleaned_query}*\n\n"
                    f"### 🔍 Looking up organization number...\n"
                    f"✓ Found organization number: `{org_id}`\n\n"
                    f"### 📥 Ingesting company data...\n"
                    f"✓ **Ingestion complete!**\n"
                    f"- Portfolio companies found: **{portfolio_count}**\n"
                    f"- Total companies processed: **{companies_processed}**"
                )
            else:
                message = (
                    f"## Company Not Found\n\n"
                    f"Organization ID: `{org_id}`\n\n"
                    f"### 📥 Ingesting new company...\n"
                    f"✓ **Ingestion complete!**\n"
                    f"- Portfolio companies found: **{portfolio_count}**\n"
                    f"- Total companies processed: **{companies_processed}**"
                )

            return AgentResponse(
                message=message,
                company_found=False,
                ingestion_triggered=True,
                ingestion_result=ingestion_result,
            )
        else:
            error_msg = ingestion_result.get("error", "Unknown error")

            if "Agent ingest is not possible" in error_msg:
                message = (
                    f"## ❌ Company Not Found\n\n**{cleaned_query}**\n\nAgent ingest is not possible, try again later."
                )
            elif query_type == "company_name":
                message = (
                    f"## ❌ Company Not Found\n\n"
                    f"**{cleaned_query}**\n\n"
                    f"Failed to ingest company:\n```\n{error_msg}\n```"
                )
            else:
                message = (
                    f"## ❌ Company Not Found\n\n"
                    f"Organization ID: `{cleaned_query}`\n\n"
                    f"Attempted ingestion but encountered an error:\n```\n{error_msg}\n```"
                )

            return AgentResponse(
                message=message,
                company_found=False,
                ingestion_triggered=True,
                ingestion_result=ingestion_result,
                error=error_msg,
            )

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return AgentResponse(
            message=f"## ⚠️ Error\n\nAn error occurred while processing your query:\n```\n{str(e)}\n```",
            company_found=False,
            error=str(e),
        )
