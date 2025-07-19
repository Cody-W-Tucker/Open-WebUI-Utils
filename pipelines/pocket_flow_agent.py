"""
title: PocketFlow Chat Agent
author: Cody W Tucker
date: 2024-12-30
version: 2.0
license: MIT
description: A clean Open-WebUI pipeline using PocketFlow's Node and Flow abstractions for multi-collection retrieval and chat.
requirements: pocketflow, openai, ollama, qdrant_client
"""

import os
import logging
from typing import List, Dict, Generator, Optional, Any, Callable
from datetime import datetime

import pydantic
from pydantic import BaseModel, Field

# PocketFlow imports
from pocketflow import Node, Flow, AsyncNode, AsyncFlow, AsyncParallelBatchNode

# Other imports
import openai
import ollama
import asyncio

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CollectionRouterNode(AsyncNode):
    """Routes user queries to relevant knowledge collections using PocketFlow AsyncNode pattern with Ollama"""
    
    async def prep_async(self, shared):
        """Extract query and available collections from shared context"""
        query = shared.get("user_query", "")
        collections = shared.get("available_collections", [])
        ollama_task_model = shared.get("ollama_task_model", "llama3.2:latest")
        return {"query": query, "collections": collections, "model": ollama_task_model}
    
    async def exec_async(self, inputs):
        """Determine relevant collections for the query using Ollama (local task model)"""
        query = inputs["query"]
        collections = inputs["collections"]
        model = inputs["model"]
        
        # Build collection descriptions
        collection_descriptions = {
            "personal": "Personal notes, thoughts, and journal entries",
            "research": "Research papers, articles, and academic content", 
            "projects": "Work projects, code, and technical documentation",
            "entities": "People, places, organizations, and other entities",
            "chat_history": "Previous conversations and interactions"
        }
        
        # Create routing prompt
        available_desc = "\n".join([
            f"- {name}: {collection_descriptions.get(name, 'Unknown collection')}" 
            for name in collections
        ])
        
        routing_prompt = f"""Given the user query, determine which knowledge collections are most relevant.

Available collections:
{available_desc}

Query: {query}

Return only the collection names as a comma-separated list (e.g., "personal,research").
For broad or unclear queries, include multiple collections.

Response:"""

        try:
            # Use Ollama for collection routing (local task model)
            # Note: ollama.chat is synchronous, but we'll run it in an executor
            def _ollama_call():
                return ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": routing_prompt}],
                    options={"temperature": 0}  # Low temperature for deterministic routing
                )
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, _ollama_call)
            
            selected_text = response['message']['content'].strip()
            selected = [name.strip() for name in selected_text.split(",")]
            # Filter to valid collection names
            valid_collections = [name for name in selected if name in collections]
            
            if not valid_collections:
                return collections  # Fallback to all collections
                
            logger.info(f"Routed query to collections: {valid_collections}")
            return valid_collections
            
        except Exception as e:
            logger.error(f"Collection routing failed: {e}")
            return collections  # Fallback to all collections
    
    async def post_async(self, shared, prep_res, exec_res):
        """Store selected collections and continue to retrieval"""
        shared["selected_collections"] = exec_res
        return "retrieve"

class DocumentRetrieverNode(AsyncParallelBatchNode):
    """Retrieves documents from selected collections in parallel using AsyncParallelBatchNode"""
    
    async def prep_async(self, shared):
        """Prepare parameters for each collection to be processed in parallel"""
        query = shared.get("user_query", "")
        collections = shared.get("selected_collections", [])
        vector_stores = shared.get("vector_stores", {})
        max_docs = shared.get("max_docs_per_collection", 3)
        
        # Return list of parameters for parallel processing
        collection_params = []
        for collection in collections:
            if collection in vector_stores:
                collection_params.append({
                    "query": query,
                    "collection": collection,
                    "vector_store": vector_stores[collection],
                    "max_docs": max_docs
                })
            else:
                logger.warning(f"Collection {collection} not available")
        
        return collection_params
    
    async def exec_async(self, collection_param):
        """Retrieve documents from a single collection (called in parallel for each collection)"""
        query = collection_param["query"]
        collection = collection_param["collection"]
        vector_store = collection_param["vector_store"]
        max_docs = collection_param["max_docs"]
        
        try:
            # Get retriever for this collection
            retriever = vector_store.as_retriever(
                search_kwargs={"k": max_docs}
            )
            
            # Retrieve documents (we need to make this async-compatible)
            def _retrieve_docs():
                return retriever.invoke(query)
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, _retrieve_docs)
            
            # Add collection metadata
            for doc in docs:
                if not hasattr(doc, 'metadata') or not doc.metadata:
                    doc.metadata = {}
                doc.metadata['collection'] = collection
                doc.metadata['retrieved_at'] = datetime.now().isoformat()
            
            logger.info(f"Retrieved {len(docs)} documents from {collection}")
            return docs
            
        except Exception as e:
            logger.error(f"Error retrieving from {collection}: {e}")
            return []  # Return empty list on error
    
    async def post_async(self, shared, prep_res, exec_res_list):
        """Combine documents from all collections and continue to processing"""
        # Flatten the list of document lists
        all_documents = []
        for docs in exec_res_list:
            all_documents.extend(docs)
        
        shared["documents"] = all_documents
        logger.info(f"Retrieved total of {len(all_documents)} documents from {len(exec_res_list)} collections")
        return "process"

class DocumentProcessorNode(AsyncNode):
    """Processes documents to extract insights using AsyncNode pattern"""
    
    async def prep_async(self, shared):
        """Extract documents from shared context"""
        documents = shared.get("documents", [])
        return documents
    
    async def exec_async(self, documents):
        """Extract insights and detect patterns in documents"""
        def _process_documents():
            insights = []
            
            if not documents:
                return insights
                
            # Group documents by collection
            collection_groups = {}
            for doc in documents:
                collection = getattr(doc, 'metadata', {}).get('collection', 'unknown')
                if collection not in collection_groups:
                    collection_groups[collection] = []
                collection_groups[collection].append(doc)
            
            # Generate insights about cross-collection patterns
            if len(collection_groups) > 1:
                insights.append(f"Found related information across {len(collection_groups)} different collections")
                
            # Check for temporal patterns
            dates = []
            for doc in documents:
                metadata = getattr(doc, 'metadata', {})
                if 'date' in metadata:
                    dates.append(metadata['date'])
            
            if len(dates) > 1:
                insights.append(f"Information spans multiple time periods ({len(dates)} dated entries)")
            
            # Basic contradiction detection
            doc_texts = [getattr(doc, 'page_content', str(doc)).lower() for doc in documents]
            contradiction_keywords = [
                ("yes", "no"), ("true", "false"), ("agree", "disagree"),
                ("positive", "negative"), ("increase", "decrease"),
                ("support", "oppose"), ("accept", "reject")
            ]
            
            for pos_word, neg_word in contradiction_keywords:
                pos_found = any(pos_word in text for text in doc_texts)
                neg_found = any(neg_word in text for text in doc_texts)
                
                if pos_found and neg_found:
                    insights.append(f"Found potential contradiction: references to both '{pos_word}' and '{neg_word}'")
                    
            return insights
        
        # Run in executor to avoid blocking if processing becomes heavy
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _process_documents)
    
    async def post_async(self, shared, prep_res, exec_res):
        """Store insights and continue to response generation"""
        shared["insights"] = exec_res
        return "generate"

class ResponseGeneratorNode(AsyncNode):
    """Generates responses using AsyncNode pattern with OpenAI"""
    
    async def prep_async(self, shared):
        """Extract all context needed for response generation"""
        return {
            "query": shared.get("user_query", ""),
            "documents": shared.get("documents", []),
            "insights": shared.get("insights", []),
            "collections": shared.get("selected_collections", [])
        }
    
    async def exec_async(self, inputs):
        """Build context for OpenAI response generation"""
        query = inputs["query"]
        documents = inputs["documents"]
        insights = inputs["insights"]
        collections = inputs["collections"]
        
        # Build context from documents
        context_text = "\n\n".join([
            f"[{getattr(doc, 'metadata', {}).get('collection', 'unknown')}] {getattr(doc, 'page_content', str(doc))}" 
            for doc in documents
        ])
        
        # Build insights section
        insights_text = "\n".join(insights) if insights else "No specific insights detected."
        
        # Create system prompt
        system_prompt = f"""You are a dynamic knowledge partner that synthesizes insights across multiple sources.

CONTEXT FROM KNOWLEDGE BASE:
{context_text}

DETECTED INSIGHTS:
{insights_text}

INSTRUCTIONS:
1. Use the provided context to answer the user's question comprehensively
2. Highlight connections and patterns between different sources
3. Point out any contradictions or tensions you notice
4. Suggest follow-up questions or areas for exploration
5. Be conversational and insightful, not just informative
6. If you notice gaps in information, acknowledge them

Remember: Your goal is to be a thought partner, not just an information retriever."""

        # Return the context for streaming (handled by the pipeline)
        return {
            "system_prompt": system_prompt,
            "query": query,
            "documents": documents
        }
    
    async def post_async(self, shared, prep_res, exec_res):
        """Store response context and end flow"""
        shared["response_context"] = exec_res
        return "default"  # End of flow

class Pipeline:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = Field(default="your-api-key-here")
        OPENAI_MODEL: str = Field(default="gpt-4o-mini-2024-07-18")
        OLLAMA_TASK_MODEL: str = Field(default="llama3.2:latest")
        OLLAMA_EMBEDDING_MODEL: str = Field(default="nomic-embed-text:latest")
        OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")
        QDRANT_URL: str = Field(default="http://localhost:6333")
        MAX_DOCS_PER_COLLECTION: int = Field(default=3)
        ENABLE_INSIGHTS: bool = Field(default=True)
        model_config = {"extra": "allow"}

    def __init__(self):
        self.name = "PocketFlow Chat Agent"
        self.vector_stores = {}
        
        # Available collections
        self.collections = ["personal", "research", "projects", "entities", "chat_history"]
        
        # Initialize PocketFlow async nodes
        self.router_node = CollectionRouterNode()
        self.retriever_node = DocumentRetrieverNode()
        self.processor_node = DocumentProcessorNode()
        self.generator_node = ResponseGeneratorNode()
        
        # Connect nodes with PocketFlow syntax
        self.router_node - "retrieve" >> self.retriever_node
        self.retriever_node - "process" >> self.processor_node
        self.processor_node - "generate" >> self.generator_node
        
        # Create the PocketFlow async flow
        self.chat_flow = AsyncFlow(start=self.router_node)
        
        # Initialize valves with environment overrides
        self.valves = self.Valves(
            OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
            OPENAI_MODEL=os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18"),
            OLLAMA_TASK_MODEL=os.getenv("OLLAMA_TASK_MODEL", "llama3.2:latest"),
            OLLAMA_EMBEDDING_MODEL=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest"),
            OLLAMA_BASE_URL=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            QDRANT_URL=os.getenv("QDRANT_URL", "http://localhost:6333"),
        )

    async def on_startup(self):
        """Initialize vector stores with Ollama embeddings"""
        try:
            logger.info("Initializing PocketFlow Chat Agent with Ollama embeddings and OpenAI chat...")
            
            # Initialize vector stores with Ollama embeddings (like original agent.py)
            from qdrant_client import QdrantClient
            
            # Create simple Ollama embeddings wrapper (like original agent.py approach)
            class OllamaEmbeddings:
                def __init__(self, model, base_url):
                    self.model = model
                    self.base_url = base_url
                
                def embed_query(self, text):
                    try:
                        response = ollama.embeddings(
                            model=self.model,
                            prompt=text
                        )
                        return response['embedding']
                    except Exception as e:
                        logger.error(f"Embedding failed: {e}")
                        return [0] * 384  # Fallback embedding
                
                def embed_documents(self, texts):
                    return [self.embed_query(text) for text in texts]
            
            embeddings = OllamaEmbeddings(
                self.valves.OLLAMA_EMBEDDING_MODEL,
                self.valves.OLLAMA_BASE_URL,
            )
            
            # Connect to Qdrant
            client = QdrantClient(
                url=self.valves.QDRANT_URL, 
                https=self.valves.QDRANT_URL.startswith("https"),
                check_compatibility=False  # Suppress version compatibility warning
            )
            
            # Initialize vector stores for available collections
            available_collections = []
            for collection in self.collections:
                try:
                    if client.collection_exists(collection):
                        # Create a simple vector store wrapper compatible with original agent.py approach
                        class QdrantVectorStore:
                            def __init__(self, client, collection_name, embedding):
                                self.client = client
                                self.collection_name = collection_name
                                self.embedding = embedding
                            
                            def as_retriever(self, search_kwargs=None):
                                return QdrantRetriever(self, search_kwargs or {})
                        
                        class QdrantRetriever:
                            def __init__(self, vector_store, search_kwargs):
                                self.vector_store = vector_store
                                self.k = search_kwargs.get("k", 3)
                            
                            def invoke(self, query):
                                try:
                                    # Get embedding for query
                                    query_embedding = self.vector_store.embedding.embed_query(query)
                                    
                                    # Search in Qdrant
                                    results = self.vector_store.client.search(
                                        collection_name=self.vector_store.collection_name,
                                        query_vector=query_embedding,
                                        limit=self.k
                                    )
                                    
                                    # Convert to document-like objects
                                    documents = []
                                    for result in results:
                                        doc = type('Document', (), {})()
                                        doc.page_content = result.payload.get('content', str(result.payload))
                                        doc.metadata = result.payload.get('metadata', {})
                                        documents.append(doc)
                                    
                                    return documents
                                    
                                except Exception as e:
                                    logger.error(f"Retrieval failed: {e}")
                                    return []  # Return empty list on error
                        
                        vector_store = QdrantVectorStore(
                            client=client,
                            collection_name=collection,
                            embedding=embeddings
                        )
                        self.vector_stores[collection] = vector_store
                        available_collections.append(collection)
                        logger.info(f"Initialized vector store for collection: {collection}")
                    else:
                        logger.warning(f"Collection {collection} does not exist")
                except Exception as e:
                    logger.error(f"Failed to initialize collection {collection}: {e}")
            
            if not available_collections:
                logger.warning("No collections are available - running in demo mode")
                available_collections = ["personal"]  # Fallback for demo
            
            self.available_collections = available_collections
            
            logger.info(f"PocketFlow Chat Agent initialized with {len(self.vector_stores)} collections")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            # Don't raise - allow pipeline to work in demo mode
            self.available_collections = ["personal"]

    async def on_shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down PocketFlow Chat Agent")

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
        __event_emitter__: Optional[Callable[[dict], None]] = None,
    ) -> Generator[str, None, None]:
        """Main processing pipeline using PocketFlow"""
        
        try:
            # Initialize shared context for PocketFlow
            shared = {
                "user_query": user_message,
                "available_collections": getattr(self, 'available_collections', ["personal"]),
                "vector_stores": getattr(self, 'vector_stores', {}),
                "ollama_task_model": self.valves.OLLAMA_TASK_MODEL,
                "ollama_base_url": self.valves.OLLAMA_BASE_URL,
                "max_docs_per_collection": self.valves.MAX_DOCS_PER_COLLECTION,
                "enable_insights": self.valves.ENABLE_INSIGHTS
            }
            
            # Run the PocketFlow async flow in sync context
            logger.info("Running PocketFlow chat flow...")
            
            async def run_async_flow():
                return await self.chat_flow.run_async(shared)
            
            # Run the async flow - handle different event loop scenarios
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If event loop is already running, we need to run in a thread
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, run_async_flow())
                        future.result()
                else:
                    loop.run_until_complete(run_async_flow())
            except RuntimeError:
                # No event loop exists, create one
                asyncio.run(run_async_flow())
            
            # Get the response context from the flow execution
            response_context = shared.get("response_context", {})
            
            if not response_context:
                error_msg = "No response generated from PocketFlow"
                logger.error(error_msg)
                if __event_emitter__:
                    __event_emitter__({"type": "stream", "data": error_msg})
                yield error_msg
                return
            
            # Generate streaming response using OpenAI
            try:
                client = openai.OpenAI(api_key=self.valves.OPENAI_API_KEY)
                
                messages_for_openai = [
                    {"role": "system", "content": response_context["system_prompt"]},
                    {"role": "user", "content": response_context["query"]}
                ]
                
                # Stream the response
                logger.info("Starting response generation with OpenAI...")
                stream = client.chat.completions.create(
                    model=self.valves.OPENAI_MODEL,
                    messages=messages_for_openai,
                    stream=True,
                    temperature=0.7
                )
                
                response_text = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        chunk_text = chunk.choices[0].delta.content
                        response_text += chunk_text
                        if __event_emitter__:
                            __event_emitter__({"type": "stream", "data": chunk_text})
                        yield chunk_text
                
                # Emit citations for retrieved documents
                documents = response_context.get("documents", [])
                if documents:
                    self._emit_citations(documents, __event_emitter__)
                
            except Exception as e:
                logger.error(f"Response generation failed: {e}")
                error_msg = f"Error generating response: {str(e)}"
                if __event_emitter__:
                    __event_emitter__({"type": "stream", "data": error_msg})
                yield error_msg
                
        except Exception as e:
            logger.error(f"PocketFlow execution failed: {e}")
            error_msg = f"Pipeline error: {str(e)}"
            if __event_emitter__:
                __event_emitter__({"type": "stream", "data": error_msg})
            yield error_msg

    def _emit_citations(self, documents: List[Any], event_emitter: Optional[Callable] = None):
        """Emit citation events for retrieved documents"""
        if not event_emitter:
            return
            
        seen_sources = set()
        
        for doc in documents:
            metadata = getattr(doc, 'metadata', {})
            source = metadata.get("source", "Unknown Source")
            collection = metadata.get("collection", "Unknown Collection")
            
            # Skip duplicates
            if source in seen_sources:
                continue
            seen_sources.add(source)
            
            # Emit citation event
            yield {
                "event": {
                    "type": "citation",
                    "data": {
                        "document": [getattr(doc, 'page_content', str(doc))],
                        "metadata": [
                            {
                                "source": source,
                                "collection": collection,
                                "retrieved_at": metadata.get("retrieved_at", datetime.now().isoformat()),
                            }
                        ],
                        "source": {
                            "name": f"[{collection}] {source}",
                        }
                    }
                }
            } 