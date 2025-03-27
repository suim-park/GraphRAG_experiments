from libs import get_entities, embed_entity, vector_search, enhanced_chunk_finder
from langchain_neo4j import Neo4jGraph
from openai import OpenAI
from typing import List, Tuple, Dict, Optional
from conn import connect2Googlesheet
import logging
import yaml
import os
from dotenv import load_dotenv


# retrieve graph context for a given query, where context is a list of entities and their relationships
def graph_retriever(graph, query: str, max_depth: int = 2) -> Dict[str, any]:
    """
    Retrieve graph context for a given query by finding relevant entities and their relationships.

    Args:
        graph: Neo4j graph connection
        query: Question or query text to find context for
        max_depth: Maximum path length for relationship traversal (default=2)

    Returns:
        Dict containing:
            - entities: List of extracted entities
            - contexts: List of relationship paths
            - metadata: Dict with retrieval statistics
    """
    try:
        # Extract entities from query
        entities, correction_context = get_entities(query)
        if not entities or entities == ["error occurred"]:
            return {"error": "Failed to extract entities from query"}

        # Get entity IDs through vector search
        entity_mappings = []
        for entity in entities:
            embedding = embed_entity(entity)
            closest_node = vector_search(graph, embedding, k=1)
            if closest_node:
                entity_mappings.append(
                    {
                        "original": entity,
                        "id": closest_node[0]["node.id"],
                        "score": closest_node[0]["score"],
                    }
                )

        # Build relationship query
        relationships = []
        for mapping in entity_mappings:
            neighbors_query = f"""
            MATCH path = (n:`__Entity__` {{id:"{mapping['id']}"}})-[r*..{max_depth}]-(m:`__Entity__`)
            WHERE ALL(rel IN relationships(path) 
                  WHERE NOT type(rel) IN ['HAS_ENTITY', 'MENTIONS'])
            RETURN 
                n.id AS source,
                n.text AS source_text,
                [rel IN relationships(path) | {{
                    type: type(rel),
                    direction: CASE 
                        WHEN startNode(rel) = n THEN "outgoing" 
                        WHEN endNode(rel) = n THEN "incoming" 
                        ELSE "undirected"
                    END
                }}] AS relations,
                [node IN nodes(path) | {{id: node.id, text: node.text}}] AS path_nodes
            LIMIT 10
            """

            result = graph.query(neighbors_query)
            for record in result:
                path_info = {
                    "source": record["source"],
                    "source_text": record["source_text"],
                    "relationships": record["relations"],
                    "path_nodes": record["path_nodes"],
                }
                relationships.append(path_info)

        # Format output
        return {
            "query": query,
            "entities": entity_mappings,
            "relationships": relationships,
            "metadata": {
                "num_entities": len(entities),
                "num_paths": len(relationships),
                "max_depth": max_depth,
            },
        }

    except Exception as e:
        logging.error(f"Error in graph_retriever: {str(e)}")
        return {"error": str(e)}


# parse graph relationships and format output as a string
def parse_relationships(context: Dict) -> str:
    """
    Parse and format relationship paths from graph context JSON.

    Args:
        context: Dictionary containing query, entities, and relationships
                Output from graph_retriever function

    Returns:
        Formatted string with directional relationships using arrow notation
    """
    if "error" in context:
        return f"Error: {context['error']}"

    if not context.get("relationships"):
        return "No relationships found."

    formatted_output = []

    # Process each relationship path
    for path in context["relationships"]:
        source_text = path["source"]
        nodes = path["path_nodes"]
        relations = path["relationships"]

        # Format each relationship in the path
        for i in range(len(relations)):
            source = source_text
            target = nodes[i + 1]["id"]
            rel_type = relations[i]["type"]
            direction = relations[i]["direction"]

            # Format with arrows based on direction
            if direction == "outgoing":
                formatted_rel = f"- {source} -> {rel_type.lower()} -> {target}"
            elif direction == "incoming":
                formatted_rel = f"- {target} -> {rel_type.lower()} -> {source}"
            else:
                formatted_rel = f"- {source} -- {rel_type.lower()} -- {target}"

            formatted_output.append(formatted_rel)

    # Remove duplicates while preserving order
    unique_rels = list(dict.fromkeys(formatted_output))

    # Add header with query information
    header = f"Query: {context['query']}\n"
    header += f"Found {len(unique_rels)} unique relationships:\n"

    return header + "\n".join(unique_rels)


# retrieve first k chunks from the graph database for a given list of filenames
def get_firstk_chunks(graph, filenames: List[str], firstk: int = 12) -> List[Dict]:
    """
    Retrieve first k chunks for each filename from the graph database.

    Args:
        graph: Neo4jGraph object
        filenames: List of filenames to search for
        firstk: Number of chunks to retrieve per file (default=5)

    Returns:
        List of dictionaries containing filename and its chunks
    """
    results = []

    for filename in filenames:
        # Add .pdf extension if not present
        if not filename.endswith(".pdf"):
            filename = f"{filename}.pdf"

        # Query to get first k chunks for the file
        query = """
        MATCH (c:Chunk)
        WHERE c.fileName = $filename AND c.position >= 5
        WITH c
        ORDER BY c.position
        LIMIT $k
        RETURN collect({
            text: c.text,
            position: c.position,
            page: c.page_number
        }) as chunks
        """

        try:
            # Execute query with parameters
            result = graph.query(query, {"filename": filename, "k": firstk})

            # Process results
            chunks = list(result)[0]["chunks"]

            if chunks:
                # Store results in required format
                file_info = {
                    "filename": filename.replace(".pdf", ""),
                    "chunk_text": [chunk["text"] for chunk in chunks],
                    "metadata": {
                        "positions": [chunk["position"] for chunk in chunks],
                        "pages": [chunk["page"] for chunk in chunks],
                    },
                }
                results.append(file_info)

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

    return results


# enhanced chunk finder function that combines vector search and graph search
def final_context_builder(graph, query, method="hybrid"):
    """
    This function performs vector search, graph search, or both to build a context string for
    an LLM

    Args:
    graph: Neo4jGraph object
    query: string

    Returns:
    context: string
    """
    final_context = ""
    if method == "vector":
        filenames, outputs = enhanced_chunk_finder(
            graph, query, limit=20, similarity_threshold=0.8, max_hops=1
        )
        final_context = (
            "Given the following context in the format [(File Name, Text),...] \n"
        )
        final_context += str([ele[:2] for ele in outputs if ele != ""])

    elif method == "graph":
        final_context = parse_relationships(graph_retriever(graph, query, max_depth=2))
        return final_context

    elif method == "hybrid":
        filenames, outputs = enhanced_chunk_finder(
            graph, query, limit=20, similarity_threshold=0.8, max_hops=1
        )
        final_context = (
            parse_relationships(graph_retriever(graph, query, max_depth=2))
            + "\n And Given the following context in the format [(File Name, Text),...] \n"
            + str([ele[:2] for ele in outputs if ele != ""])
        )
    else:
        pass  # no context
    return final_context, filenames


# load system prompt template from YAML file
def load_prompt_template(prompt_type: str) -> str:
    """
    Load prompt template from YAML file.

    Args:
        prompt_type: Type of prompt to load (e.g., 'document_retrieval', 'medical_synthesis')

    Returns:
        str: Loaded prompt template
    """
    try:
        with open("sysprompts.yaml", "r") as file:
            prompts = yaml.safe_load(file)
            return prompts.get(prompt_type, {}).get("system", "")
    except Exception as e:
        logging.error(f"Error loading prompt template: {e}")
        return ""


# synthesize response using OpenAI API
def synthesize_response(
    graph, query, method="hybrid", synth_type="stance_synthesis", model="gpt-4"
) -> Optional[Dict]:
    """...existing docstring..."""

    final_context, filenames = final_context_builder(graph, query, method)
    system_prompt = load_prompt_template(synth_type)
    firstk_chunks_prompt = get_firstk_chunks(graph, filenames, firstk=10)

    prompt = f"""{system_prompt}
    
    User Query: {query}
    Context: {final_context}
    """

    load_dotenv()
    # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "system", "content": str(firstk_chunks_prompt)},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
        )
        return response

    except Exception as e:
        print(f"Error during API call: {e}")
        return None, None


# Function to compute metrics per question
def compute_per_question_metrics(df):
    stance_categories = ["Support", "Against", "Neutral"]
    results = []

    for idx, row in df.iterrows():
        q_num = row["Question Number"]
        q_metrics = {"Question Number": q_num}

        for stance in stance_categories:
            pred_str = row.get(f"{stance}_x", "")
            true_str = row.get(f"{stance}_y", "")

            pred_set = set([x.strip() for x in pred_str.split(",") if x.strip()])
            true_set = set([x.strip() for x in true_str.split(",") if x.strip()])

            universe = pred_set.union(true_set)
            TP = len(pred_set & true_set)
            FP = len(pred_set - true_set)
            FN = len(true_set - pred_set)
            TN = len(universe) - (TP + FP + FN)

            acc = (TP + TN) / len(universe) if len(universe) else 0
            prec = TP / (TP + FP) if (TP + FP) else 0
            rec = TP / (TP + FN) if (TP + FN) else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

            q_metrics[f"{stance}_Accuracy"] = acc
            q_metrics[f"{stance}_Precision"] = prec
            q_metrics[f"{stance}_Recall"] = rec
            q_metrics[f"{stance}_F1"] = f1

        results.append(q_metrics)

    return pd.DataFrame(results)


if __name__ == "__main__":  # Fixed syntax error (== instead of =)
    # Import required packages
    from langchain_neo4j import Neo4jGraph
    import os
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Initialize Neo4j connection
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URL"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
    )

    # Test case 1: Basic query with different methods
    test_query = "What is the effect of dexmedetomidine on delirium?"
    print("=== Testing Basic Query ===")

    # Test hybrid method
    print("\nTesting Hybrid Method:")
    response_hybrid = synthesize_response(
        graph=graph,
        query=test_query,
        method="hybrid",
        synth_type="stance_synthesis",
        model="gpt-4",
    )
    if response_hybrid and hasattr(response_hybrid, "choices"):
        print("Hybrid Response:", response_hybrid.choices[0].message.content)

    # Test vector method
    print("\nTesting Vector Method:")
    response_vector = synthesize_response(
        graph=graph,
        query=test_query,
        method="vector",
        synth_type="stance_synthesis",
        model="gpt-3.5-turbo",
    )
    if response_vector and hasattr(response_vector, "choices"):
        print("Vector Response:", response_vector.choices[0].message.content)

    # Test graph retriever function
    print("\nTesting Graph Retriever:")
    graph_context = graph_retriever(graph, test_query, max_depth=2)
    print("Graph Context:", graph_context.get("metadata", {}))

    # Test first k chunks retrieval
    print("\nTesting First K Chunks:")
    test_files = ["ACURASYS", "ROSE"]
    chunks = get_firstk_chunks(graph, test_files, firstk=3)
    for file_info in chunks:
        print(f"\nFile: {file_info['filename']}")
        print(f"Number of chunks: {len(file_info['chunk_text'])}")
