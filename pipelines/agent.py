"""
title: Context-Rich Exploratory RAG Chatbot
author: Cody W Tucker
date: 2024-12-30
version: 1.0
license: MIT
description: A pipeline for retrieving and synthesizing information across multiple knowledge bases.
requirements: langchain==0.3.3, langchain_core==0.3.10, langchain_openai==0.3.18, openai==1.82.0, langchain_qdrant==0.2.0, qdrant_client==1.11.0, pydantic==2.7.4
"""

import os
import logging
from typing import List, Dict, Generator, Optional, Any
import pydantic
import hashlib
from datetime import datetime
print(f"Loaded Pydantic version: {pydantic.__version__}")
print(f"Pydantic module path: {pydantic.__file__}")

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever

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
        OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"
        SYSTEM_PROMPT: str
        MAX_DOCUMENTS_PER_COLLECTION: int = 5
        model_config = {"extra": "allow"}

    def __init__(self):
        self.name = "Agent"
        self.vector_stores = {}  # Dict mapping collection name to vector store
        self.retrievers = {}  # Dict mapping collection name to retriever
        self.llm = None
        self.embeddings = None
        
        # Hardcoded collection values
        self.qdrant_urls = ["http://qdrant.homehub.tv"]
        self.qdrant_collections = ["personal", "chat_history", "research", "projects", "entities"]
        
        # Hardcoded collection descriptions
        self.collection_descriptions = {
            "personal": "Personal knowledge base with notes, journal entries, and thoughts.",
            "chat_history": "Chat history with the user.",
            "research": "Research papers, articles, and reference materials.",
            "projects": "Current ideas and initiatives that I am working on.",
            "entities": "List of People, Places, and Things."
        }
        
        # Hardcoded ensemble weights
        self.ensemble_weights = {
            "personal": 0.3,
            "chat_history": 0.2,
            "research": 0.1,
            "projects": 0.1,
            "entities": 0.3
        }
        
        self.valves = self.Valves(**{
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "your-api-key-here"),
            "TASK_OPENAI_MODEL": os.getenv("TASK_OPENAI_MODEL", "gpt-4o-mini-2024-07-18"),
            "LARGE_OPENAI_MODEL": os.getenv("LARGE_OPENAI_MODEL", "gpt-4o-2024-11-20"),
            "OPENAI_EMBEDDING_MODEL": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
            "SYSTEM_PROMPT": os.getenv("SYSTEM_PROMPT", 
                """You are a dynamic knowledge partner capable of synthesizing insights across multiple sources.
                Use the provided context to not just answer questions, but to highlight connections, 
                identify patterns, and suggest new perspectives. When appropriate, note contradictions
                or tensions between different sources. If you don't know the answer, acknowledge that
                and suggest alternative approaches."""),
            "MAX_DOCUMENTS_PER_COLLECTION": int(os.getenv("MAX_DOCUMENTS_PER_COLLECTION", "5")),
        })

    async def on_startup(self):
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        
        try:
            logger.info("Starting initialization...")
            self.embeddings = OpenAIEmbeddings(
                model=self.valves.OPENAI_EMBEDDING_MODEL, 
                api_key=self.valves.OPENAI_API_KEY
            )
            logger.info(f"Embeddings initialized with model {self.valves.OPENAI_EMBEDDING_MODEL}")
            
            # Print the collections we're trying to connect to
            logger.info(f"Attempting to connect to collections: {', '.join(self.qdrant_collections)}")
            logger.info(f"Using Qdrant URLs: {', '.join(self.qdrant_urls)}")
            
            # Create a client for each URL
            clients = {}
            for url in self.qdrant_urls:
                try:
                    clients[url] = QdrantClient(url=url, https=url.startswith("https"))
                    logger.info(f"Connected to Qdrant at {url}")
                except Exception as e:
                    logger.error(f"Failed to connect to Qdrant at {url}: {str(e)}")
            
            if not clients:
                raise ValueError("Could not connect to any Qdrant servers")
            
            # Initialize each vector store
            # Check each collection on each client
            for collection in self.qdrant_collections:
                connected = False
                for url, client in clients.items():
                    try:
                        if client.collection_exists(collection):
                            logger.info(f"Collection '{collection}' exists at {url}")
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
                                    if hasattr(self.retriever, "aget_relevant_documents"):
                                        docs = await self.retriever.aget_relevant_documents(query)
                                    else:
                                        # Fall back to sync if needed
                                        docs = self.retriever.get_relevant_documents(query)
                                    
                                    # Add collection metadata
                                    for doc in docs:
                                        if isinstance(doc.metadata, dict):
                                            doc.metadata["collection"] = self.collection
                                        else:
                                            doc.metadata = {"collection": self.collection}
                                    return docs
                            
                            # Wrap with metadata transformer and add to retrievers dictionary
                            self.retrievers[collection] = CollectionMetadataRetriever(base_retriever, collection)
                            
                            logger.info(f"Vector store and retriever for {collection} initialized")
                            connected = True
                            break
                        else:
                            logger.warning(f"Collection '{collection}' does not exist at {url}")
                    except Exception as e:
                        logger.error(f"Error checking collection '{collection}' at {url}: {str(e)}")
                
                if not connected:
                    logger.warning(f"⚠️ Could not connect to collection '{collection}' on any server")
            
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
            
        except Exception as e:
            logger.error(f"Failed to initialize: {str(e)}")
            raise

    async def on_shutdown(self):
        logger.info("Shutting down Context-Rich Exploratory RAG Pipeline")
        # Clean up any resources if needed

    async def _determine_relevant_collections(self, user_message: str, chat_history: List[dict]) -> List[str]:
        """Determine which collections are most relevant for the query using LLM."""
        # If LLM is not initialized, fall back to all collections
        if not self.llm:
            logger.warning("LLM not initialized, using all collections")
            return list(self.vector_stores.keys())
        
        # Extract recent chat context to help with collection selection
        recent_context = ""
        if chat_history and len(chat_history) > 0:
            # Get last 2 exchanges
            recent_messages = chat_history[-4:] if len(chat_history) >= 4 else chat_history
            for msg in recent_messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role and content:
                    recent_context += f"{role.capitalize()}: {content}\n"
            
        # Define a more structured prompt for the LLM
        prompt = f"""You are a query router that determines which knowledge collections are most relevant for answering a user's query.

                AVAILABLE COLLECTIONS:
                """
        
        # Add descriptions of each collection with examples of when they would be relevant
        for collection, description in self.collection_descriptions.items():
            if collection in self.vector_stores:
                prompt += f"- {collection}: {description}\n"
        
        # Add recent conversation context if available
        if recent_context:
            prompt += "\nRECENT CONVERSATION CONTEXT:\n" + recent_context + "\n"
            
        # Add examples to clarify when each collection is relevant
        prompt += """
            EXAMPLES OF COLLECTION SELECTION:
            1. Query: "What did I write in my journal last week?"
            Relevant collections: personal
            Reason: This is asking about personal journal entries.

            2. Query: "What was our conversation yesterday about machine learning?"
            Relevant collections: chat_history, research
            Reason: This is explicitly asking about a previous conversation and research on machine learning.

            3. Query: "Tell me about recent research in quantum computing"
            Relevant collections: research
            Reason: This is asking for research information on a scientific topic.

            4. Query: "What are some programming techniques I've used before?"
            Relevant collections: projects, chat_history
            Reason: This could be found in both project notes and previous conversations.

            5. Query: "Who is John Smith?"
            Relevant collections: entities
            Reason: John is a person listed in the entities collection.

            USER QUERY: {query}

            INSTRUCTIONS:
            - Analyze if the query refers to personal information, past conversations, or research topics
            - Choose ONLY the collections that are MOST relevant for answering this specific query
            - If the query mentions a topic, or concept that might be in multiple collections, include all relevant ones
            - Research doesn't contain people, so if the query mentions a person, it should be in the entities collection and possibly the chat_history collection
            - Return ONLY a comma-separated list of collection names, nothing else
            - If you're uncertain which is best, return 'all'

            RELEVANT COLLECTIONS:""".format(query=user_message)
                    
        try:
            # Create a non-streaming version of the LLM for this specific task
            from langchain_openai import ChatOpenAI
            task_llm = ChatOpenAI(
                model=self.valves.TASK_OPENAI_MODEL,
                api_key=self.valves.OPENAI_API_KEY,
                temperature=0,  # Use low temperature for deterministic outputs
                streaming=False,
            )
            
            # Call the LLM
            response = await task_llm.ainvoke(prompt)
            collections_text = response.content.strip()
            
            logger.info(f"LLM response for collection selection: {collections_text}")
            
            # Parse the response
            if collections_text.lower() == 'all':
                logger.info("LLM suggested using all collections")
                return list(self.vector_stores.keys())
            
            # Split by comma and strip whitespace
            selected_collections = [col.strip() for col in collections_text.split(',')]
            
            # Filter to only collections that exist in our vector stores
            valid_collections = [col for col in selected_collections if col in self.vector_stores]
            
            if not valid_collections:
                logger.warning("LLM didn't return any valid collections, using all collections")
                return list(self.vector_stores.keys())
                
            logger.info(f"Selected collections based on LLM: {valid_collections}")
            return valid_collections
            
        except Exception as e:
            logger.error(f"Error using LLM for collection selection: {str(e)}", exc_info=True)
            logger.info("Falling back to all collections")
            return list(self.vector_stores.keys())

    def _format_document_metadata(self, doc: Document) -> Dict[str, Any]:
        """Extract and format metadata from a document."""
        metadata = doc.metadata if hasattr(doc, "metadata") else {}
        result = {
            "source": metadata.get("source", "Unknown Source"),
            "collection": metadata.get("collection", "Unknown Collection"),
            "date": metadata.get("date", "Unknown Date"),
        }
        
        # Add any other metadata that might be useful
        for key, value in metadata.items():
            if key not in result:
                result[key] = value
                
        return result

    def _create_document_hash(self, doc: Document) -> str:
        """Create a hash of document content for deduplication."""
        # Use a combination of content and key metadata
        content = doc.page_content if hasattr(doc, "page_content") else ""
        metadata = doc.metadata if hasattr(doc, "metadata") else {}
        
        # Include source in hash to avoid deduping different documents with similar content
        source = metadata.get("source", "")
        
        # Create hash
        hash_input = f"{content}|{source}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """Remove duplicate documents based on content hash."""
        unique_docs = []
        seen_hashes = set()
        
        for doc in documents:
            doc_hash = self._create_document_hash(doc)
            if doc_hash not in seen_hashes:
                seen_hashes.add(doc_hash)
                unique_docs.append(doc)
                
        logger.info(f"Deduplicated {len(documents)} documents to {len(unique_docs)} unique documents")
        return unique_docs

    def _create_contextualized_ensemble_retriever(self, relevant_collections: List[str], messages: List[dict]) -> Optional[EnsembleRetriever]:
        """Create an EnsembleRetriever from retrievers for the relevant collections."""
        if not relevant_collections:
            return None
            
        # Filter to only collections we have retrievers for
        available_collections = [col for col in relevant_collections if col in self.retrievers]
        logger.info(f"Available collections for retrieval: {available_collections}")
        logger.debug(f"All configured collections: {self.qdrant_collections}")
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
            collection_weights = {col: self.ensemble_weights.get(col, 0.33) for col in available_collections}
            
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
            ensemble = EnsembleRetriever(
                retrievers=ensemble_retrievers,
                weights=retriever_weights
            )
            return ensemble
        return None

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Generator[str, None, None]:
        # Log the actual user message we're processing
        logger.info(f"Processing query: '{user_message}'")
        
        # Determine which collections to query - call async function from sync context
        try:
            import asyncio
            relevant_collections = asyncio.run(self._determine_relevant_collections(user_message, messages))
        except RuntimeError:
            # RuntimeError might occur if there's already an event loop running
            # Fall back to all collections
            logger.warning("Could not run async collection determination, using all collections")
            relevant_collections = list(self.vector_stores.keys())
            
        if not isinstance(relevant_collections, list):
            logger.warning(f"Collection determination returned non-list: {type(relevant_collections)}")
            relevant_collections = list(self.vector_stores.keys())
            
        logger.info(f"Relevant collections determined: {', '.join(relevant_collections)}")
        
        # Create an ensemble retriever with retrievers
        ensemble_retriever = self._create_contextualized_ensemble_retriever(relevant_collections, messages)
        
        # If we couldn't create an ensemble retriever, respond accordingly
        if not ensemble_retriever:
            logger.error("Could not create ensemble retriever")
            yield "I couldn't access any knowledge bases. Please try again later or contact support."
            return
            
        # Retrieve documents using the ensemble retriever
        try:
            logger.info(f"Using EnsembleRetriever to retrieve documents for: '{user_message}'")
            
            # Use the ensemble retriever directly
            try:
                # Retrieve documents
                all_documents = ensemble_retriever.get_relevant_documents(user_message)
                
                # Make sure collection metadata is present
                for doc in all_documents:
                    if not doc.metadata.get("collection"):
                        # If we couldn't determine, use "unknown"
                        doc.metadata["collection"] = "unknown"
            except Exception as e:
                logger.error(f"Error in ensemble retriever: {str(e)}", exc_info=True)
                logger.info("Falling back to retrieving from each collection separately")
                
                # Fallback: retrieve from each collection separately
                all_documents = []
                for collection in relevant_collections:
                    if collection in self.retrievers:
                        try:
                            retriever = self.retrievers[collection]
                            docs = retriever.get_relevant_documents(user_message)
                            all_documents.extend(docs)
                            logger.info(f"Retrieved {len(docs)} documents from {collection}")
                        except Exception as e:
                            logger.error(f"Error retrieving from {collection}: {str(e)}", exc_info=True)
            
            # Deduplicate documents
            all_documents = self._deduplicate_documents(all_documents)
            
            # Print summary of retrieved documents
            collections_found = {}
            for doc in all_documents:
                collection = doc.metadata.get("collection", "unknown")
                collections_found[collection] = collections_found.get(collection, 0) + 1
                
            logger.info(f"Retrieved {len(all_documents)} total documents from collections: {collections_found}")
            
            # Sort documents by collection weight
            def get_collection_weight(doc):
                collection = doc.metadata.get("collection", "unknown")
                return self.ensemble_weights.get(collection, 0.33)
            
            # Sort documents by collection weight (highest first)
            all_documents.sort(key=get_collection_weight, reverse=True)
            
            # Limit total documents to avoid overloading the LLM
            max_docs = 15
            if len(all_documents) > max_docs:
                logger.info(f"Limiting from {len(all_documents)} to {max_docs} documents")
                
                # Group documents by collection
                docs_by_collection = {}
                for doc in all_documents:
                    coll = doc.metadata.get('collection', 'unknown')
                    if coll not in docs_by_collection:
                        docs_by_collection[coll] = []
                    docs_by_collection[coll].append(doc)
                
                # If multiple collections, keep a balanced set
                if len(docs_by_collection) > 1:
                    # Take documents from each collection proportional to their weights
                    balanced_docs = []
                    
                    # Calculate how many to take from each collection based on weights
                    total_weight = sum(self.ensemble_weights.get(coll, 0.33) for coll in docs_by_collection.keys())
                    for coll, docs in docs_by_collection.items():
                        weight = self.ensemble_weights.get(coll, 0.33)
                        proportion = weight / total_weight
                        num_docs = max(1, min(len(docs), round(proportion * max_docs)))
                        balanced_docs.extend(docs[:num_docs])
                    
                    # If we still have space, add more from highest weighted collections
                    remaining = max_docs - len(balanced_docs)
                    if remaining > 0:
                        # Sort collections by weight
                        sorted_colls = sorted(docs_by_collection.keys(), 
                                            key=lambda c: self.ensemble_weights.get(c, 0.33),
                                            reverse=True)
                        
                        for coll in sorted_colls:
                            if remaining <= 0:
                                break
                            docs = docs_by_collection[coll]
                            used = len([d for d in balanced_docs if d.metadata.get('collection') == coll])
                            if used < len(docs):
                                additional = min(remaining, len(docs) - used)
                                balanced_docs.extend(docs[used:used+additional])
                                remaining -= additional
                    
                    all_documents = balanced_docs
                else:
                    # Just take the top max_docs
                    all_documents = all_documents[:max_docs]
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}", exc_info=True)
            yield f"Error retrieving information: {str(e)}"
            return
        
        # If no documents were retrieved, respond accordingly
        if not all_documents:
            logger.warning("No documents were retrieved from any collection")
            yield "I couldn't find any relevant information in my knowledge base. Could you provide more details or try a different question?"
            return
            
        # Make sure we have content in the documents
        has_content = False
        content_counts = {}
        for doc in all_documents:
            collection = doc.metadata.get("collection", "unknown")
            if doc.page_content and len(doc.page_content.strip()) > 0:
                has_content = True
                content_counts[collection] = content_counts.get(collection, 0) + 1
                
        logger.info(f"Content counts by collection: {content_counts}")
                
        if not has_content:
            logger.warning("Documents were retrieved but they have no content")
            yield "I found documents but they don't contain useful information. Please try a different query."
            return
        
        # Prepare system prompt with relationship and contradiction information
        system_prompt = self.valves.SYSTEM_PROMPT
        
        # Print the documents we're using
        logger.info(f"Using {len(all_documents)} total documents for answer generation")
        
        # Clean up system prompt and add the context tag
        system_prompt = system_prompt.rstrip() + "\n\n{context}"
        
        # Create QA prompt
        qa_prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
        )
        
        # Create QA chain
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        # Instead of using create_retrieval_chain with a lambda, directly pass documents to QA chain
        try:
            logger.info("Starting answer generation")
            
            response_stream = question_answer_chain.stream({
                "input": user_message,
                "chat_history": messages,
                "context": all_documents
            })
            logger.info("Got response stream")
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}", exc_info=True)
            yield f"Error: {str(e)}"
            return
            
        # Track if we've received an answer
        has_answer = False
        answer = ""
        
        # Stream the answer chunks
        for chunk in response_stream:
            # Handle different response formats
            if isinstance(chunk, str):
                # Direct string response
                has_answer = True
                answer += chunk
                yield chunk
            elif isinstance(chunk, dict):
                # Dictionary response (typical for chain output)
                if "answer" in chunk:
                    has_answer = True
                    answer_chunk = chunk["answer"]
                    answer += answer_chunk
                    yield answer_chunk
        
        # If no answer was generated from the model
        if not has_answer:
            logger.warning("No answer was generated from the model")
            yield "I don't know."
            return
            
        # Append citations if documents were retrieved
        if all_documents:
            # Send citations as events
            seen_sources = set()  # To avoid duplicates
            for doc in all_documents:
                # Extract metadata
                metadata = doc.metadata if hasattr(doc, "metadata") else {}
                source = metadata.get("source", "Unknown Source")
                collection = metadata.get("collection", "Unknown Collection")
                
                # Skip duplicates (optional) - using a set to track seen sources
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
            
            # Final newline for spacing
            yield "\n"
