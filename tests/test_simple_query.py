from app.db.neo4j_client import get_driver
from typing import List, Optional, Dict, Any
import logging
import re

logger = logging.getLogger(__name__)


def upsert_company(company_data: Dict[str, Any]) -> None:
    """
    Upsert a Company node with full metadata.
    
    Validates that company_id is a valid Swedish organization number before upserting.
    """
    company_id = company_data.get("company_id", "")
    
    # Validate organization number format (10 digits)
    cleaned = re.sub(r'[-\s]', '', str(company_id))
    if not (len(cleaned) == 10 and cleaned.isdigit()):
        raise ValueError(f"Invalid organization number format: '{company_id}'. Must be 10 digits (format: XXXXXX-XXXX)")
    
    query = """
    MERGE (c:Company {company_id: $company_id})
    SET c.name = $name,
        c.country_code = $country_code,
        c.description = $description,
        c.mission = $mission,
        c.sectors = $sectors,
        c.updated_at = datetime(),
        
        // New fields supported by LLM ingestion
        c.website = $website,
        c.num_employees = $num_employees,
        c.year_founded = $year_founded,
        c.aliases = $aliases,
        c.key_people = $key_people

    WITH c
    WHERE $cluster_id IS NOT NULL
    SET c.cluster_id = $cluster_id
    WITH c
    WHERE $vector IS NOT NULL
    SET c.vector = $vector
    WITH c
    SET c.portfolio = COALESCE($portfolio, [])
    """

    # Prepare params with defaults for safety
    params = {
        "company_id": company_data["company_id"],
        "name": company_data["name"],
        "country_code": company_data.get("country_code", "SE"),
        "description": company_data.get("description", ""),
        "mission": company_data.get("mission", ""),
        "sectors": company_data.get("sectors", []),
        "website": company_data.get("website", ""),
        "num_employees": company_data.get("num_employees"),
        "year_founded": str(company_data.get("year_founded") or ""),
        "aliases": company_data.get("aliases", []),
        "key_people": company_data.get("key_people", []),
        "cluster_id": company_data.get("cluster_id"),
        "vector": company_data.get("vector"),
        "portfolio": company_data.get("portfolio"),
    }

    driver = get_driver()
    with driver.session() as session:
        session.run(query, params)


def get_company(company_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a Company or Fund node by company_id.
    """
    query = """
    MATCH (n)
    WHERE (n:Company OR n:Fund)
      AND n.company_id = $company_id
    RETURN n
    """
    driver = get_driver()
    with driver.session() as session:
        result = session.run(query, company_id=company_id)
        record = result.single()
        if record:
            node_data = dict(record["n"])
            logger.debug(f"get_company({company_id}) found: name={repr(node_data.get('name'))}, keys={list(node_data.keys())}")
            return node_data
        logger.debug(f"get_company({company_id}) returned None")
        return None


def find_company_by_name(company_name: str) -> Optional[Dict[str, Any]]:
    """
    Find a Company or Fund node by name (case-insensitive, handles variations).
    
    FIXED: Handles BOTH patterns in your database:
    1. Companies where name = actual company name (e.g., "Volvo")
    2. Companies where name = org ID (e.g., "556012-5790")
    
    Searches in this order:
    1. Exact match on name field
    2. Exact match on aliases
    3. Fuzzy match on name field
    4. Fuzzy match on aliases
    """
    # Normalize company name for matching
    normalized_name = company_name.strip().lower()
    # Remove common suffixes for better matching
    normalized_name_clean = normalized_name.replace(" ab", "").replace(" ab publ", "").replace(" ab (publ)", "").replace(" ltd", "").strip()
    
    logger.info(f"🔍 Searching for company: '{company_name}' -> normalized: '{normalized_name}'")
    
    driver = get_driver()
    
    # STRATEGY 1: Exact match on name field (case-insensitive)
    # This handles both "Volvo" and org IDs stored in name field
    query = """
    MATCH (n)
    WHERE (n:Company OR n:Fund)
      AND n.name IS NOT NULL
      AND toLower(n.name) = $normalized_name
    RETURN n
    LIMIT 1
    """
    with driver.session() as session:
        result = session.run(query, normalized_name=normalized_name)
        record = result.single()
        if record:
            node_data = dict(record["n"])
            logger.info(f"✓ Found exact name match: {node_data.get('name')} (ID: {node_data.get('company_id')})")
            return node_data
        logger.debug(f"No exact match on name field for '{normalized_name}'")
    
    # STRATEGY 2: Try cleaned name (without suffixes like "AB", "Ltd")
    if normalized_name_clean != normalized_name:
        query = """
        MATCH (n)
        WHERE (n:Company OR n:Fund)
          AND n.name IS NOT NULL
          AND toLower(n.name) = $normalized_name_clean
        RETURN n
        LIMIT 1
        """
        with driver.session() as session:
            result = session.run(query, normalized_name_clean=normalized_name_clean)
            record = result.single()
            if record:
                node_data = dict(record["n"])
                logger.info(f"✓ Found cleaned name match: {node_data.get('name')} (ID: {node_data.get('company_id')})")
                return node_data
            logger.debug(f"No cleaned match for '{normalized_name_clean}'")
    
    # STRATEGY 3: Exact match on aliases array
    query = """
    MATCH (n)
    WHERE (n:Company OR n:Fund)
      AND n.aliases IS NOT NULL
      AND SIZE(n.aliases) > 0
      AND ANY(alias IN n.aliases 
          WHERE alias IS NOT NULL 
          AND toLower(alias) = $normalized_name)
    RETURN n
    LIMIT 1
    """
    with driver.session() as session:
        result = session.run(query, normalized_name=normalized_name)
        record = result.single()
        if record:
            node_data = dict(record["n"])
            logger.info(f"✓ Found alias exact match: {node_data.get('name')} (ID: {node_data.get('company_id')})")
            return node_data
        logger.debug(f"No exact alias match for '{normalized_name}'")
    
    # STRATEGY 4: Fuzzy match on name - CONTAINS
    query = """
    MATCH (n)
    WHERE (n:Company OR n:Fund)
      AND n.name IS NOT NULL
      AND (
        toLower(n.name) CONTAINS $normalized_name
        OR $normalized_name CONTAINS toLower(n.name)
      )
    RETURN n
    LIMIT 1
    """
    with driver.session() as session:
        result = session.run(query, normalized_name=normalized_name)
        record = result.single()
        if record:
            node_data = dict(record["n"])
            logger.info(f"✓ Found fuzzy name match: {node_data.get('name')} (ID: {node_data.get('company_id')})")
            return node_data
        logger.debug(f"No fuzzy name match for '{normalized_name}'")
    
    # STRATEGY 5: Fuzzy match with cleaned name
    if normalized_name_clean != normalized_name:
        query = """
        MATCH (n)
        WHERE (n:Company OR n:Fund)
          AND n.name IS NOT NULL
          AND (
            toLower(n.name) CONTAINS $normalized_name_clean
            OR $normalized_name_clean CONTAINS toLower(n.name)
          )
        RETURN n
        LIMIT 1
        """
        with driver.session() as session:
            result = session.run(query, normalized_name_clean=normalized_name_clean)
            record = result.single()
            if record:
                node_data = dict(record["n"])
                logger.info(f"✓ Found fuzzy cleaned match: {node_data.get('name')} (ID: {node_data.get('company_id')})")
                return node_data
            logger.debug(f"No fuzzy cleaned match for '{normalized_name_clean}'")
    
    # STRATEGY 6: Fuzzy match on aliases
    query = """
    MATCH (n)
    WHERE (n:Company OR n:Fund)
      AND n.aliases IS NOT NULL
      AND SIZE(n.aliases) > 0
      AND ANY(alias IN n.aliases 
          WHERE alias IS NOT NULL 
          AND (
            toLower(alias) CONTAINS $normalized_name
            OR $normalized_name CONTAINS toLower(alias)
          ))
    RETURN n
    LIMIT 1
    """
    with driver.session() as session:
        result = session.run(query, normalized_name=normalized_name)
        record = result.single()
        if record:
            node_data = dict(record["n"])
            logger.info(f"✓ Found fuzzy alias match: {node_data.get('name')} (ID: {node_data.get('company_id')})")
            return node_data
        logger.debug(f"No fuzzy alias match for '{normalized_name}'")
    
    logger.warning(f"❌ No match found for '{company_name}' after trying all 6 strategies")
    return None


def convert_company_to_fund(company_id: str) -> None:
    """
    Convert a Company node to Fund by removing Company label and adding Fund label.
    """
    query = """
    MATCH (n {company_id: $company_id})
    WHERE n:Company
    REMOVE n:Company
    SET n:Fund
    """
    driver = get_driver()
    with driver.session() as session:
        session.run(query, company_id=company_id)


def search_similar_companies(
    vector: List[float], limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Find similar companies using vector search.
    Assumes a vector index exists on :Company(vector).
    """
    query = """
    CALL db.index.vector.queryNodes('company_vector_index', $limit, $vector)
    YIELD node, score
    RETURN node, score
    """

    driver = get_driver()
    with driver.session() as session:
        result = session.run(query, vector=vector, limit=limit)
        return [
            {"company": dict(record["node"]), "score": record["score"]}
            for record in result
        ]


def get_companies_by_cluster(cluster_id: int) -> List[Dict[str, Any]]:
    """
    Get all companies in a specific cluster.
    """
    query = """
    MATCH (c:Company {cluster_id: $cluster_id})
    RETURN c
    """
    driver = get_driver()
    with driver.session() as session:
        result = session.run(query, cluster_id=cluster_id)
        return [dict(record["c"]) for record in result]


# ========================================
# DEBUG UTILITIES
# ========================================

def debug_search_company(company_name: str) -> None:
    """
    Debug function to see ALL matching strategies for a company name.
    Shows which strategies find matches and what they return.
    """
    normalized_name = company_name.strip().lower()
    normalized_name_clean = normalized_name.replace(" ab", "").replace(" ltd", "").strip()
    
    print(f"\n{'='*60}")
    print(f"DEBUG SEARCH FOR: '{company_name}'")
    print(f"Normalized: '{normalized_name}'")
    print(f"Cleaned: '{normalized_name_clean}'")
    print(f"{'='*60}\n")
    
    driver = get_driver()
    
    strategies = [
        ("Exact name match", """
            MATCH (n)
            WHERE (n:Company OR n:Fund)
              AND n.name IS NOT NULL
              AND toLower(n.name) = $normalized_name
            RETURN n.name as name, n.company_id as id, labels(n) as labels
        """, {"normalized_name": normalized_name}),
        
        ("Cleaned name match", """
            MATCH (n)
            WHERE (n:Company OR n:Fund)
              AND n.name IS NOT NULL
              AND toLower(n.name) = $normalized_name_clean
            RETURN n.name as name, n.company_id as id, labels(n) as labels
        """, {"normalized_name_clean": normalized_name_clean}),
        
        ("Exact alias match", """
            MATCH (n)
            WHERE (n:Company OR n:Fund)
              AND n.aliases IS NOT NULL
              AND ANY(alias IN n.aliases WHERE toLower(alias) = $normalized_name)
            RETURN n.name as name, n.company_id as id, labels(n) as labels, n.aliases as aliases
        """, {"normalized_name": normalized_name}),
        
        ("Fuzzy name CONTAINS", """
            MATCH (n)
            WHERE (n:Company OR n:Fund)
              AND n.name IS NOT NULL
              AND toLower(n.name) CONTAINS $normalized_name
            RETURN n.name as name, n.company_id as id, labels(n) as labels
            LIMIT 3
        """, {"normalized_name": normalized_name}),
        
        ("Fuzzy alias CONTAINS", """
            MATCH (n)
            WHERE (n:Company OR n:Fund)
              AND n.aliases IS NOT NULL
              AND ANY(alias IN n.aliases WHERE toLower(alias) CONTAINS $normalized_name)
            RETURN n.name as name, n.company_id as id, labels(n) as labels, n.aliases as aliases
            LIMIT 3
        """, {"normalized_name": normalized_name}),
    ]
    
    for strategy_name, query, params in strategies:
        print(f"\n{'─'*60}")
        print(f"Strategy: {strategy_name}")
        print(f"{'─'*60}")
        
        with driver.session() as session:
            result = session.run(query, params)
            records = list(result)
            
            if records:
                print(f"✓ Found {len(records)} match(es):")
                for i, rec in enumerate(records, 1):
                    print(f"  {i}. Name: {rec.get('name')}")
                    print(f"     ID: {rec.get('id')}")
                    print(f"     Labels: {rec.get('labels')}")
                    if rec.get('aliases'):
                        print(f"     Aliases: {rec.get('aliases')}")
            else:
                print(f"✗ No matches")
    
    print(f"\n{'='*60}\n")


# Example usage
if __name__ == "__main__":
    # Test with the problematic searches
    debug_search_company("ABB Ltd")
    debug_search_company("Volvo")
    debug_search_company("200601-9240")  # Direct org ID search
