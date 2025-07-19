"""
title: Context-Rich Exploratory RAG Chatbot
author: Cody W Tucker
date: 2024-12-30
version: 1.0
license: MIT
description: A pipeline for retrieving and synthesizing information across multiple knowledge bases.
requirements: langchain==0.3.3, langchain_core==0.3.10, langchain_openai==0.3.18, openai==1.82.0, langchain_qdrant==0.2.0, qdrant_client==1.11.0, pydantic==2.7.4, langchain_ollama
"""

import os
import logging
from typing import List, Dict, Generator, Optional, Any, Callable
import pydantic

from datetime import datetime
print(f"Loaded Pydantic version: {pydantic.__version__}")
print(f"Pydantic module path: {pydantic.__file__}")

from pydantic import BaseModel, Field
from langchain.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Set up logging - avoid duplicate handlers
logger = logging.getLogger(__name__)
# Remove all existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
# Add our handler
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Determine log level from environment variable
# In a production environment, set PIPELINE_LOG_LEVEL to WARNING or ERROR for efficiency.
# Defaults to ERROR if not set or if an invalid value is provided.
_log_level_str = os.getenv("PIPELINE_LOG_LEVEL", "ERROR").upper()
_numeric_level = getattr(logging, _log_level_str, None)

if not isinstance(_numeric_level, int):
    # Use a print statement here for initial setup, as logger's level isn't set yet.
    print(f"Warning: Invalid log level '{_log_level_str}' from PIPELINE_LOG_LEVEL. Defaulting to INFO.")
    _numeric_level = logging.INFO

logger.setLevel(_numeric_level)
# This message will only appear if the configured level is INFO or DEBUG.
logger.info(f"Logger initialized for '{__name__}'. Effective log level: {logging.getLevelName(logger.getEffectiveLevel())}")

class Pipeline:
    class Valves(BaseModel):
        OPENAI_API_KEY: str
        TASK_OPENAI_MODEL: str = "gpt-4o-mini-2024-07-18"
        LARGE_OPENAI_MODEL: str = "gpt-4o-2024-11-20"
        OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text:latest"
        OLLAMA_BASE_URL: str = "http://localhost:11434"
        SYSTEM_PROMPT: str
        MAX_DOCUMENTS_PER_COLLECTION: int = 5
        QDRANT_URL: str
        model_config = {"extra": "allow"}

    def __init__(self):
        self.name = "Agent"
        self.vector_stores = {}  # Dict mapping collection name to vector store
        self.retrievers = {}  # Dict mapping collection name to retriever
        self.llm = None
        self.task_llm = None  # Add task_llm instance variable
        self.embeddings = None
        self.citation = False  # Disable Open WebUI's built-in citations

        self.QDRANT_URL = None
        self.QDRANT_COLLECTIONS: List[str] = [
            "personal",
            "chat_history",
            "research",
            "projects",
            "entities",
        ]
        self.COLLECTION_DESCRIPTIONS: Dict[str, str] = {
            "personal": "Personal knowledge base with notes, journal entries, and thoughts.",
            "chat_history": "Chat history with the user.",
            "research": "Research papers, articles, and reference materials.",
            "projects": "Current ideas and initiatives that I am working on.",
            "entities": "List of People, Places, and Things."
        }
        self.ENSEMBLE_WEIGHTS: Dict[str, float] = {
            "personal": 0.3,
            "chat_history": 0.2,
            "research": 0.1,
            "projects": 0.1,
            "entities": 0.3
        }
        
        # Initialize valves with environment variable overrides where applicable
        self.valves = self.Valves(
            OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
            TASK_OPENAI_MODEL=os.getenv("TASK_OPENAI_MODEL", "gpt-4o-mini-2024-07-18"),
            LARGE_OPENAI_MODEL=os.getenv("LARGE_OPENAI_MODEL", "gpt-4o-2024-11-20"),
            OLLAMA_EMBEDDING_MODEL=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest"),
            OLLAMA_BASE_URL=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            SYSTEM_PROMPT=os.getenv("SYSTEM_PROMPT", 
                """You are a dynamic knowledge partner capable of synthesizing insights across multiple sources.
                Use the provided context to not just answer questions, but to highlight connections, 
                identify patterns, and suggest new perspectives. When appropriate, note contradictions
                or tensions between different sources. If you don't know the answer, acknowledge that
                and suggest alternative approaches."""
            ),
            QDRANT_URL=os.getenv("QDRANT_URL", "http://localhost:6333")
        )

    def _convert_to_lc_messages(self, chat_history_dicts: List[dict]) -> List[Any]: # Using Any for BaseMessage for simplicity here
        lc_msgs = []
        for msg in chat_history_dicts:
            role = msg.get("role")
            content = msg.get("content")
            if role == "user":
                lc_msgs.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_msgs.append(AIMessage(content=content))
            # System messages are usually not part of a history like this,
            # but could be added if your chat_history_dicts include them.
        return lc_msgs

    async def on_startup(self):
        from langchain_openai import ChatOpenAI
        from langchain_ollama import OllamaEmbeddings
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        
        try:
            logger.info("Starting initialization...")
            self.embeddings = OllamaEmbeddings(
                model=self.valves.OLLAMA_EMBEDDING_MODEL,
                base_url=self.valves.OLLAMA_BASE_URL,
            )
            logger.info(f"Embeddings initialized with Ollama model {self.valves.OLLAMA_EMBEDDING_MODEL} at {self.valves.OLLAMA_BASE_URL}")
            
            # Print the collections we're trying to connect to
            logger.info(f"Attempting to connect to collections: {', '.join(self.QDRANT_COLLECTIONS)}")
            logger.info(f"Using Qdrant URL: {self.valves.QDRANT_URL}")
            
            # Create a single client with better debugging
            logger.info(f"Attempting to connect to Qdrant URL: {self.valves.QDRANT_URL}")
            logger.info(f"URL starts with https: {self.valves.QDRANT_URL.startswith('https')}")
            
            # Test basic network connectivity
            import urllib.parse
            parsed_url = urllib.parse.urlparse(self.valves.QDRANT_URL)
            logger.info(f"Parsed URL - scheme: {parsed_url.scheme}, netloc: {parsed_url.netloc}, path: {parsed_url.path}")
            
            # Try to ping the host to see if it's reachable
            import socket
            try:
                host = parsed_url.netloc.split(':')[0] if ':' in parsed_url.netloc else parsed_url.netloc
                port = int(parsed_url.netloc.split(':')[1]) if ':' in parsed_url.netloc else (443 if parsed_url.scheme == 'https' else 80)
                logger.info(f"Testing connectivity to {host}:{port}")
                
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((host, port))
                sock.close()
                
                if result == 0:
                    logger.info(f"Network connectivity test passed for {host}:{port}")
                else:
                    logger.warning(f"Network connectivity test failed for {host}:{port} (error code: {result})")
            except Exception as net_e:
                logger.warning(f"Network connectivity test failed: {str(net_e)}")
            
            try:
                # Try the standard connection first
                client = QdrantClient(url=self.valves.QDRANT_URL, https=self.valves.QDRANT_URL.startswith("https"))
                logger.info(f"Successfully connected to Qdrant at {self.valves.QDRANT_URL}")
                
                # Test the connection by getting server info
                try:
                    collections_info = client.get_collections()
                    logger.info(f"Connection test successful. Server responded with collections info.")
                except Exception as test_e:
                    logger.warning(f"Connected but couldn't get collections info: {str(test_e)}")
                    
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant at {self.valves.QDRANT_URL}: {str(e)}")
                
                # Try alternative connection methods
                logger.info("Trying alternative connection methods...")
                
                # Try without https flag
                try:
                    logger.info("Trying without https flag...")
                    client = QdrantClient(url=self.valves.QDRANT_URL)
                    logger.info(f"Alternative connection successful: {self.valves.QDRANT_URL}")
                except Exception as e2:
                    logger.error(f"Alternative connection failed: {str(e2)}")
                    
                    # Try with explicit https=True
                    try:
                        logger.info("Trying with explicit https=True...")
                        client = QdrantClient(url=self.valves.QDRANT_URL, https=True)
                        logger.info(f"HTTPS connection successful: {self.valves.QDRANT_URL}")
                    except Exception as e3:
                        logger.error(f"HTTPS connection also failed: {str(e3)}")
                        raise ValueError(f"Could not connect to Qdrant server at {self.valves.QDRANT_URL}. All connection methods failed.")
            
            # Initialize each vector store
            # Check each collection on the client
            for collection in self.QDRANT_COLLECTIONS:
                try:
                    if client.collection_exists(collection):
                        logger.info(f"Collection '{collection}' exists")
                        vector_store = QdrantVectorStore(
                            client=client,
                            collection_name=collection,
                            embedding=self.embeddings
                        )
                        self.vector_stores[collection] = vector_store
                        
                        # Create a retriever for this collection
                        base_retriever = vector_store.as_retriever(
                            search_kwargs={"k": self.valves.MAX_DOCUMENTS_PER_COLLECTION}
                        )
                        
                        # Create a retriever that adds collection metadata to documents
                        class CollectionMetadataRetriever(BaseRetriever):
                            """A proper BaseRetriever implementation that adds collection metadata to documents."""
                            retriever: Any = Field(description="The retriever to delegate to")
                            collection: str = Field(description="The collection name")
                            
                            def __init__(self, retriever, collection_name):
                                super().__init__(retriever=retriever, collection=collection_name)
                            
                            def _get_relevant_documents(self, query, **kwargs):
                                # Use invoke() if retriever supports it, otherwise fall back to get_relevant_documents
                                if hasattr(self.retriever, "invoke"):
                                    docs = self.retriever.invoke(query)
                                else:
                                    docs = self.retriever.get_relevant_documents(query)
                                
                                # Add collection metadata
                                for doc in docs:
                                    if isinstance(doc.metadata, dict):
                                        doc.metadata["collection"] = self.collection
                                    else:
                                        doc.metadata = {"collection": self.collection}
                                return docs
                            
                            async def _aget_relevant_documents(self, query, **kwargs):
                                """Async implementation for better performance."""
                                # Try to use invoke() first, which is the preferred LangChain method
                                if hasattr(self.retriever, "ainvoke"):
                                    docs = await self.retriever.ainvoke(query)
                                elif hasattr(self.retriever, "aget_relevant_documents"):
                                    docs = await self.retriever.aget_relevant_documents(query)
                                elif hasattr(self.retriever, "invoke"):
                                    docs = self.retriever.invoke(query)
                                else:
                                    # Fall back to sync if needed
                                    docs = self.retriever.get_relevant_documents(query)
                                
                                # Add collection metadata
                                for doc in docs:
                                    if not hasattr(doc, 'metadata') or not doc.metadata:
                                        doc.metadata = {"collection": self.collection}
                                    else:
                                        doc.metadata["collection"] = self.collection
                                return docs
                        
                        # Wrap with metadata transformer and add to retrievers dictionary
                        self.retrievers[collection] = CollectionMetadataRetriever(base_retriever, collection)
                        
                        logger.info(f"Vector store and retriever for {collection} initialized")
                    else:
                        logger.warning(f"Collection '{collection}' does not exist")
                except Exception as e:
                    logger.error(f"Error checking collection '{collection}': {str(e)}")
            
            if not self.vector_stores:
                raise ValueError("No valid vector stores could be initialized")
            else:
                logger.info(f"Successfully initialized {len(self.vector_stores)} vector stores: {', '.join(self.vector_stores.keys())}")
            
            self.llm = ChatOpenAI(
                model=self.valves.LARGE_OPENAI_MODEL,
                api_key=self.valves.OPENAI_API_KEY,
                streaming=True,
            )
            logger.info(f"LLM initialized with model {self.valves.LARGE_OPENAI_MODEL}")
            
            # Initialize task_llm for collection routing
            self.task_llm = ChatOpenAI(
                model=self.valves.TASK_OPENAI_MODEL,
                api_key=self.valves.OPENAI_API_KEY,
                temperature=0,  # Use low temperature for deterministic outputs
                streaming=False,
            )
            logger.info(f"Task LLM initialized with model {self.valves.TASK_OPENAI_MODEL}")

        except Exception as e:
            logger.error(f"Failed to initialize: {str(e)}")
            raise

    async def on_shutdown(self):
        logger.info("Shutting down Context-Rich Exploratory RAG Pipeline")
        # Clean up any resources if needed

    async def _determine_relevant_collections(self, user_message: str) -> List[str]:
        """Determine which collections are most relevant for the query using LLM."""
        # If LLM is not initialized, return all collections
        if not self.llm or not self.task_llm:
            logger.warning("LLM not initialized, using all collections")
            return list(self.vector_stores.keys())
        
        # Build collection descriptions string
        collection_details = [f"- {col}: {desc}" for col, desc in self.COLLECTION_DESCRIPTIONS.items() if col in self.vector_stores]
        collections_string = "\n".join(collection_details)

        # Define structured prompt
        system_message_content = f"""
        You are a query router that determines which knowledge collections are most relevant for answering a user's query.

        AVAILABLE COLLECTIONS:
        {collections_string}

        INSTRUCTIONS:
        - For queries about specific people (e.g., "Who is John Smith?", "Tell me about Cody Tucker"), return ONLY "entities".
        - For queries about past conversations or chat logs, include "chat_history".
        - For queries about research, scientific, or technical topics, include "research".
        - For queries about personal notes, diary, or journal entries, include "personal".
        - For queries about coding, software development, or current projects, include "projects".
        - If a query spans multiple topics, select ALL relevant collections based on the query's intent.
        - For vague or ambiguous queries (e.g., "Tell me something interesting"), select ALL collections.
        - Do NOT select irrelevant collections (e.g., do NOT select "personal" for queries about people unless explicitly about journal entries).
        - Return a comma-separated list of collection names, nothing else.

        EXAMPLES OF COLLECTION SELECTION:
        1. Query: "What did I write in my journal last week?"
        Relevant collections: personal
        Reason: This is asking about personal journal entries.

        2. Query: "What was our conversation yesterday about machine learning?"
        Relevant collections: chat_history,research
        Reason: This asks about a past conversation and a technical topic.

        3. Query: "Tell me about recent research in quantum computing"
        Relevant collections: research
        Reason: This is asking for research information on a scientific topic.

        4. Query: "What are some programming techniques I've used before?"
        Relevant collections: projects,chat_history
        Reason: This could be found in project notes or past conversations.

        5. Query: "Who is John Smith?"
        Relevant collections: entities
        Reason: This is asking about a specific person.

        6. Query: "Tell me something about my recent projects and who worked on them"
        Relevant collections: projects,entities
        Reason: This involves project details and specific people.

        7. Query: "What's new in my life?"
        Relevant collections: personal,chat_history
        Reason: This is vague but likely involves personal notes or recent conversations.
        """

        # Define human message content
        human_template_content = f"""

        USER QUERY: {user_message}

        RELEVANT COLLECTION OR COLLECTIONS:
        """

        chat_prompt = ChatPromptTemplate.from_messages([
                ("system", system_message_content),
                ("user", human_template_content + "\n{user_message}"),
            ])

        try:
            response = await self.task_llm.ainvoke(
                chat_prompt.format_prompt(user_message=user_message).to_messages()
            )
            collections_text = response.content.strip()

            # Validate response format
            if not collections_text or collections_text.lower() in ['none', '']:
                logger.warning("LLM returned empty or invalid response, using all collections")
                return list(self.vector_stores.keys())

            # Handle 'all' explicitly
            if collections_text.lower() == 'all':
                logger.info("LLM suggested using all collections")
                return list(self.vector_stores.keys())

            # Normalize and split
            collections_text = ' '.join(collections_text.split())
            collections_text = collections_text.replace('，', ',').replace(';', ',').replace(' and ', ',')
            selected_collections = [
                col.strip().lower() for col in collections_text.split(',')
                if col.strip() and col.strip().lower() in self.vector_stores
            ]

            # If no valid collections, use all collections
            if not selected_collections:
                logger.warning(f"No valid collections in LLM response: {collections_text}, using all collections")
                return list(self.vector_stores.keys())

            logger.info(f"Query: '{user_message}', Selected collections: {selected_collections}, Reason: LLM response='{collections_text}'")
            return selected_collections

        except Exception as e:
            logger.error(f"Error using LLM for collection selection: {str(e)}", exc_info=True)
            logger.info("Using all collections due to error")
            return list(self.vector_stores.keys())

    def _create_contextualized_ensemble_retriever(self, relevant_collections: List[str], messages: List[dict]) -> Optional[EnsembleRetriever]:
        """Create an EnsembleRetriever from retrievers for the relevant collections."""
        if not relevant_collections:
            return None
            
        # Filter to only collections we have retrievers for
        available_collections = [col for col in relevant_collections if col in self.retrievers]
        logger.info(f"Available collections for retrieval: {available_collections}")
        logger.debug(f"All configured collections: {self.QDRANT_COLLECTIONS}")
        logger.debug(f"Collections with retrievers: {list(self.retrievers.keys())}")
        
        if not available_collections:
            return None
            
        # Use the BaseRetriever instances directly
        ensemble_retrievers = []
        retriever_weights = []
        
        # Calculate dynamic weights based on selected collections
        # If only one collection is selected, give it full weight
        if len(available_collections) == 1:
            collection_weights = {available_collections[0]: 1.0}
        else:
            # Start with default weights
            collection_weights = {col: self.ENSEMBLE_WEIGHTS.get(col, 0.33) for col in available_collections}
            
            # Normalize weights to sum to 1
            total_weight = sum(collection_weights.values())
            if total_weight > 0:
                collection_weights = {col: weight/total_weight for col, weight in collection_weights.items()}
        
        # Process each collection
        for collection in available_collections:
            # Get our retriever for this collection (already a BaseRetriever)
            retriever = self.retrievers[collection]
            
            # Add the retriever to our list
            ensemble_retrievers.append(retriever)
            
            # Get weight for this collection
            weight = collection_weights.get(collection, 0.33)
            logger.info(f"Using weight {weight:.2f} for collection {collection}")
            retriever_weights.append(weight)
            
        # Create and return the ensemble retriever
        if ensemble_retrievers:
            logger.info(f"Creating ensemble retriever with {len(ensemble_retrievers)} retrievers")
            
            # Create a custom EnsembleRetriever with enhanced metadata handling
            class EnhancedEnsembleRetriever(EnsembleRetriever):
                def _get_relevant_documents(self, query, **kwargs):
                    # Get documents from all retrievers
                    all_docs = []
                    for i, retriever in enumerate(self.retrievers):
                        # Use invoke() if available, otherwise fall back to get_relevant_documents
                        if hasattr(retriever, "invoke"):
                            docs = retriever.invoke(query)
                        else:
                            docs = retriever.get_relevant_documents(query)
                        logger.debug(f"Retriever {i} returned {len(docs)} documents")
                        # Verify metadata is present
                        for j, doc in enumerate(docs):
                            if not hasattr(doc, 'metadata') or not doc.metadata:
                                logger.warning(f"Document {j} from retriever {i} has no metadata! Adding empty dict.")
                                doc.metadata = {}
                            if 'collection' not in doc.metadata:
                                logger.warning(f"Document {j} from retriever {i} has no collection in metadata! Collection unknown.")
                        all_docs.append(docs)
                    
                    # Weight and merge documents
                    results = []
                    for docs, weight in zip(all_docs, self.weights):
                        for doc in docs:
                            # Explicitly add weight to metadata
                            doc.metadata["weight"] = weight
                            results.append(doc)
                    
                    # Log final results
                    logger.info(f"Enhanced ensemble retriever returning {len(results)} documents")
                    for i, doc in enumerate(results):
                        logger.debug(f"Final document {i} metadata: {doc.metadata}")
                    
                    return results
                
                async def _aget_relevant_documents(self, query, **kwargs):
                    """Async implementation for better performance."""
                    # Get documents from all retrievers
                    all_docs = []
                    for i, retriever in enumerate(self.retrievers):
                        # Try different methods in order of preference
                        if hasattr(retriever, "ainvoke"):
                            docs = await retriever.ainvoke(query)
                        elif hasattr(retriever, "aget_relevant_documents"):
                            docs = await retriever.aget_relevant_documents(query)
                        elif hasattr(retriever, "invoke"):
                            docs = retriever.invoke(query)
                        else:
                            docs = retriever.get_relevant_documents(query)
                            
                        logger.debug(f"Retriever {i} returned {len(docs)} documents")
                        # Verify metadata is present
                        for j, doc in enumerate(docs):
                            if not hasattr(doc, 'metadata') or not doc.metadata:
                                logger.warning(f"Document {j} from retriever {i} has no metadata! Adding empty dict.")
                                doc.metadata = {}
                            if 'collection' not in doc.metadata:
                                logger.warning(f"Document {j} from retriever {i} has no collection in metadata! Collection unknown.")
                        all_docs.append(docs)
                    
                    # Weight and merge documents
                    results = []
                    for docs, weight in zip(all_docs, self.weights):
                        for doc in docs:
                            # Explicitly add weight to metadata
                            doc.metadata["weight"] = weight
                            results.append(doc)
                    
                    # Log final results
                    logger.info(f"Enhanced ensemble retriever returning {len(results)} documents")
                    for i, doc in enumerate(results):
                        logger.debug(f"Final document {i} metadata: {doc.metadata}")
                    
                    return results
            
            # Use our enhanced retriever
            ensemble = EnhancedEnsembleRetriever(
                retrievers=ensemble_retrievers,
                weights=retriever_weights
            )
            return ensemble
        return None

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
        __event_emitter__: Optional[Callable[[dict], None]] = None,
    ) -> Generator[Any, None, None]:
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        logger.info(f"Processing query: '{user_message}'")
        
        try:
            # Create an event loop if one doesn't exist
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async collection determination
            relevant_collections = loop.run_until_complete(
                self._determine_relevant_collections(user_message)
            )
        except Exception as e: 
            logger.error(f"Could not run collection determination due to {str(e)}", exc_info=True)
            logger.info("Using all collections due to error")
            relevant_collections = list(self.vector_stores.keys())
            
        if not isinstance(relevant_collections, list) or not relevant_collections:
            logger.warning(f"Collection determination returned invalid result: {relevant_collections}. Using all vector stores.")
            relevant_collections = list(self.vector_stores.keys())
            if not relevant_collections:
                logger.error("No vector stores available even after fallback.")
                if __event_emitter__:
                    __event_emitter__({"type": "stream", "data": "I couldn't access any knowledge bases. Please try again later or contact support."})
                yield "I couldn't access any knowledge bases. Please try again later or contact support."
                return
            
        logger.info(f"Relevant collections determined: {', '.join(relevant_collections)}")
        
        ensemble_retriever = self._create_contextualized_ensemble_retriever(relevant_collections, messages)
        
        if not ensemble_retriever:
            logger.error("Could not create ensemble retriever")
            if __event_emitter__:
                __event_emitter__({"type": "stream", "data": "I couldn't access any knowledge bases. Please try again later or contact support."})
            yield "I couldn't access any knowledge bases. Please try again later or contact support."
            return

        # Retrieve relevant documents
        try:
            logger.info(f"Retrieving documents for query: '{user_message}'")
            # Use invoke() if available, otherwise fall back to get_relevant_documents
            if hasattr(ensemble_retriever, "invoke"):
                docs = ensemble_retriever.invoke(user_message)
            else:
                docs = ensemble_retriever.get_relevant_documents(user_message)
            logger.info(f"Retrieved {len(docs)} documents")
            
            # Log document details for debugging
            for i, doc in enumerate(docs):
                logger.info(f"Document {i} content length: {len(doc.page_content)}")
                logger.info(f"Document {i} metadata: {doc.metadata}")
                # Check if required metadata fields exist
                if not hasattr(doc, 'metadata') or not doc.metadata:
                    logger.warning(f"Document {i} has no metadata!")
                elif 'source' not in doc.metadata:
                    logger.warning(f"Document {i} has no source in metadata!")
                elif 'collection' not in doc.metadata:
                    logger.warning(f"Document {i} has no collection in metadata!")
            
            # Build context from documents
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Define the system prompt with context
            system_prompt = self.valves.SYSTEM_PROMPT + "\n\nUse the following context to answer the question:\n\n" + context
            
            # Direct chat with OpenAI
            # Create a fresh streaming LLM instance
            streaming_llm = ChatOpenAI(
                model=self.valves.LARGE_OPENAI_MODEL,
                api_key=self.valves.OPENAI_API_KEY,
                streaming=True,
            )
            
            # Create a simple prompt
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="history"),
                HumanMessage(content="{question}"),
            ])
            
            # Convert main messages to LangChain messages for history
            history_lc_messages = self._convert_to_lc_messages(messages)
            
            # Create a simple chain
            chain = prompt | streaming_llm
            
            # Stream the response
            logger.info("Starting to stream response")
            answer = ""
            for chunk in chain.stream({"question": user_message, "history": history_lc_messages}):
                # ChatOpenAI with streaming=True returns chunks with a .content attribute
                if hasattr(chunk, 'content'):
                    chunk_text = chunk.content
                    answer += chunk_text
                    if __event_emitter__:
                        __event_emitter__({"type": "stream", "data": chunk_text})
                    yield chunk_text
            
            # After streaming the answer, emit citations
            if docs:
                logger.info(f"Emitting citations for {len(docs)} source documents")
                seen_sources = set()  # To avoid duplicates
                for doc in docs:
                    # Extract metadata
                    metadata = doc.metadata if hasattr(doc, "metadata") else {}
                    source = metadata.get("source", "Unknown Source")
                    collection = metadata.get("collection", "Unknown Collection")
                    
                    # Skip duplicates
                    if source in seen_sources:
                        continue
                    seen_sources.add(source)
                    
                    yield {
                        "event": {
                            "type": "citation",
                            "data": {
                                "document": [doc.page_content],
                                "metadata": [
                                    {
                                        "source": source,
                                        "collection": collection,
                                        "date_accessed": datetime.now().isoformat(),
                                    }
                                ],
                                "source": {
                                    "name": source,
                                }
                            }
                        }
                    }

            yield "\n"

        except Exception as e:
            logger.error(f"Error during chain execution: {str(e)}", exc_info=True)
            if __event_emitter__:
                __event_emitter__({"type": "stream", "data": f"Error processing your request: {str(e)}"})
            yield f"Error processing your request: {str(e)}"
            return
