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
from typing import List, Dict, Union, Generator, Iterator, Tuple, Optional, Any
import pydantic
import sys
import urllib.parse
from datetime import datetime
print(f"Loaded Pydantic version: {pydantic.__version__}")
print(f"Pydantic module path: {pydantic.__file__}")

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever

class Pipeline:
    class Valves(BaseModel):
        OPENAI_API_KEY: str
        OPENAI_MODEL: str = "gpt-4o-mini-2024-07-18"
        OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"
        SYSTEM_PROMPT: str
        OBSIDIAN_VAULT_NAME: str
        MAX_DOCUMENTS_PER_COLLECTION: int = 5
        model_config = {"extra": "allow"}

    def __init__(self):
        self.name = "Context-Rich Exploratory RAG"
        self.vector_stores = {}  # Dict mapping collection name to vector store
        self.retrievers = {}  # Dict mapping collection name to retriever
        self.llm = None
        self.embeddings = None
        
        # Hardcoded collection values
        self.qdrant_urls = ["http://qdrant.homehub.tv"]
        self.qdrant_collections = ["personal", "chat_history", "research"]
        
        # Hardcoded collection descriptions
        self.collection_descriptions = {
            "personal": "Personal knowledge base with notes, journal entries, and thoughts",
            "chat_history": "Chat history with the user",
            "research": "Research papers, articles, and reference materials"
        }
        
        # Hardcoded ensemble weights
        self.ensemble_weights = {
            "personal": 0.7,
            "chat_history": 0.2,
            "research": 0.1
        }
        
        self.valves = self.Valves(**{
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "your-api-key-here"),
            "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18"),
            "OPENAI_EMBEDDING_MODEL": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
            "SYSTEM_PROMPT": os.getenv("SYSTEM_PROMPT", 
                """You are a dynamic knowledge partner capable of synthesizing insights across multiple sources.
                Use the provided context to not just answer questions, but to highlight connections, 
                identify patterns, and suggest new perspectives. When appropriate, note contradictions
                or tensions between different sources. If you don't know the answer, acknowledge that
                and suggest alternative approaches."""),
            "OBSIDIAN_VAULT_NAME": os.getenv("OBSIDIAN_VAULT_NAME", "MyVault"),
            "MAX_DOCUMENTS_PER_COLLECTION": int(os.getenv("MAX_DOCUMENTS_PER_COLLECTION", "5")),
        })

    async def on_startup(self):
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        
        try:
            print("Starting initialization...")
            self.embeddings = OpenAIEmbeddings(
                model=self.valves.OPENAI_EMBEDDING_MODEL, 
                api_key=self.valves.OPENAI_API_KEY
            )
            print(f"Embeddings initialized with model {self.valves.OPENAI_EMBEDDING_MODEL}")
            
            # Print the collections we're trying to connect to
            print(f"Attempting to connect to collections: {', '.join(self.qdrant_collections)}")
            print(f"Using Qdrant URLs: {', '.join(self.qdrant_urls)}")
            
            # Create a client for each URL
            clients = {}
            for url in self.qdrant_urls:
                try:
                    clients[url] = QdrantClient(url=url, https=url.startswith("https"))
                    print(f"Connected to Qdrant at {url}")
                except Exception as e:
                    print(f"Failed to connect to Qdrant at {url}: {str(e)}")
            
            if not clients:
                raise ValueError("Could not connect to any Qdrant servers")
            
            # Initialize each vector store
            # Check each collection on each client
            for collection in self.qdrant_collections:
                connected = False
                for url, client in clients.items():
                    try:
                        if client.collection_exists(collection):
                            print(f"Collection '{collection}' exists at {url}")
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
                            class CollectionMetadataTransformer:
                                """A simple wrapper to add collection metadata to documents."""
                                def __init__(self, retriever, collection_name):
                                    self.retriever = retriever
                                    self.collection = collection_name
                                
                                def get_relevant_documents(self, query):
                                    docs = self.retriever.get_relevant_documents(query)
                                    # Add collection metadata
                                    for doc in docs:
                                        if isinstance(doc.metadata, dict):
                                            doc.metadata["collection"] = self.collection
                                        else:
                                            doc.metadata = {"collection": self.collection}
                                    return docs
                            
                            # Wrap with metadata transformer and add to retrievers dictionary
                            self.retrievers[collection] = CollectionMetadataTransformer(base_retriever, collection)
                            
                            print(f"Vector store and retriever for {collection} initialized")
                            connected = True
                            break
                        else:
                            print(f"Collection '{collection}' does not exist at {url}")
                    except Exception as e:
                        print(f"Error checking collection '{collection}' at {url}: {str(e)}")
                
                if not connected:
                    print(f"⚠️ WARNING: Could not connect to collection '{collection}' on any server")
            
            if not self.vector_stores:
                raise ValueError("No valid vector stores could be initialized")
            else:
                print(f"Successfully initialized {len(self.vector_stores)} vector stores: {', '.join(self.vector_stores.keys())}")
            
            self.llm = ChatOpenAI(
                model=self.valves.OPENAI_MODEL,
                api_key=self.valves.OPENAI_API_KEY,
                streaming=True,
            )
            print(f"LLM initialized with model {self.valves.OPENAI_MODEL}")
            
        except Exception as e:
            print(f"Failed to initialize: {str(e)}")
            raise

    async def on_shutdown(self):
        print("Shutting down Context-Rich Exploratory RAG Pipeline")
        # Clean up any resources if needed

    def _determine_relevant_collections(self, user_message: str, chat_history: List[dict]) -> List[str]:
        """Determine which collections are most relevant for the query."""
        # Simple implementation - use all collections for now
        # In a more advanced implementation, we could use the LLM to determine relevant collections
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

    def _create_contextualized_ensemble_retriever(self, relevant_collections: List[str], messages: List[dict]) -> Optional[EnsembleRetriever]:
        """Create an EnsembleRetriever from retrievers for the relevant collections."""
        if not relevant_collections:
            return None
            
        # Filter to only collections we have retrievers for
        available_collections = [col for col in relevant_collections if col in self.retrievers]
        print(f"Available collections for retrieval: {available_collections}")
        print(f"All configured collections: {self.qdrant_collections}")
        print(f"Collections with retrievers: {list(self.retrievers.keys())}")
        
        if not available_collections:
            return None
            
        # Convert our custom retrievers to a format compatible with EnsembleRetriever
        # We need to create retrievers that implement BaseRetriever but delegate to our custom retrievers
        ensemble_retrievers = []
        retriever_weights = []
        
        # Process each collection
        for collection in available_collections:
            # Get our custom retriever for this collection
            custom_retriever = self.retrievers[collection]
            
            # Create a proper BaseRetriever implementation that delegates to our custom retriever
            class CompatibleRetriever(BaseRetriever):
                delegate: Any = Field(description="The retriever to delegate to")
                collection_name: str = Field(description="The name of the collection")
                
                def __init__(self, delegate_retriever, collection_name):
                    super().__init__(delegate=delegate_retriever, collection_name=collection_name)
                
                def _get_relevant_documents(self, query, **kwargs):
                    # Just delegate to our custom retriever
                    docs = self.delegate.get_relevant_documents(query)
                    # Double-check collection metadata
                    for doc in docs:
                        if not doc.metadata.get("collection"):
                            doc.metadata["collection"] = self.collection_name
                    return docs
                    
            # Create a compatible retriever
            compatible_retriever = CompatibleRetriever(custom_retriever, collection)
                
            # Add the compatible retriever to our list
            ensemble_retrievers.append(compatible_retriever)
            
            # Get weight for this collection
            weight = self.ensemble_weights.get(collection, 0.33)
            print(f"Using weight {weight} for collection {collection}")
            retriever_weights.append(weight)
            
        # Create and return the ensemble retriever
        if ensemble_retrievers:
            print(f"Creating ensemble retriever with {len(ensemble_retrievers)} retrievers")
            ensemble = EnsembleRetriever(
                retrievers=ensemble_retrievers,
                weights=retriever_weights
            )
            return ensemble
        return None

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Generator[str, None, None]:
        # Determine which collections to query
        relevant_collections = self._determine_relevant_collections(user_message, messages)
        print(f"Relevant collections determined: {', '.join(relevant_collections)}")
        
        # Create an ensemble retriever with retrievers
        ensemble_retriever = self._create_contextualized_ensemble_retriever(relevant_collections, messages)
        
        # If we couldn't create an ensemble retriever, respond accordingly
        if not ensemble_retriever:
            print("Could not create ensemble retriever")
            yield "I couldn't access any knowledge bases. Please try again later or contact support."
            return
            
        # Retrieve documents using the ensemble retriever
        try:
            print(f"Using EnsembleRetriever to retrieve documents for: '{user_message}'")
            
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
                print(f"Error in ensemble retriever: {str(e)}")
                print("Falling back to retrieving from each collection separately")
                
                # Fallback: retrieve from each collection separately
                all_documents = []
                for collection in relevant_collections:
                    if collection in self.retrievers:
                        try:
                            retriever = self.retrievers[collection]
                            docs = retriever.get_relevant_documents(user_message)
                            all_documents.extend(docs)
                            print(f"Retrieved {len(docs)} documents from {collection}")
                        except Exception as e:
                            print(f"Error retrieving from {collection}: {str(e)}")
            
            # Print summary of retrieved documents
            collections_found = {}
            for doc in all_documents:
                collection = doc.metadata.get("collection", "unknown")
                collections_found[collection] = collections_found.get(collection, 0) + 1
                
            print(f"Retrieved {len(all_documents)} total documents from collections: {collections_found}")
            
            # Sort documents by collection weight
            def get_collection_weight(doc):
                collection = doc.metadata.get("collection", "unknown")
                return self.ensemble_weights.get(collection, 0.33)
            
            # Sort documents by collection weight (highest first)
            all_documents.sort(key=get_collection_weight, reverse=True)
            
            # Limit total documents to avoid overloading the LLM
            max_docs = 15
            if len(all_documents) > max_docs:
                print(f"Limiting from {len(all_documents)} to {max_docs} documents")
                
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
            print(f"Error retrieving documents: {str(e)}")
            yield f"Error retrieving information: {str(e)}"
            return
        
        # If no documents were retrieved, respond accordingly
        if not all_documents:
            print("No documents were retrieved from any collection")
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
                
        print(f"Content counts by collection: {content_counts}")
                
        if not has_content:
            print("Documents were retrieved but they have no content")
            yield "I found documents but they don't contain useful information. Please try a different query."
            return
        
        # Prepare system prompt with relationship and contradiction information
        system_prompt = self.valves.SYSTEM_PROMPT
        
        # Print the documents we're using
        print(f"Using {len(all_documents)} total documents for answer generation")
        
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
            print("Starting answer generation")
            
            response_stream = question_answer_chain.stream({
                "input": user_message,
                "chat_history": messages,
                "context": all_documents
            })
            print("Got response stream")
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
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
        
        # If no answer was streamed, yield a fallback
        if not has_answer:
            print("No answer was generated from the model")
            yield "I don't know."
            return
            
        # Append citations if documents were retrieved
        if all_documents:
            # More compact header with proper spacing
            yield "\n"
            
            # Collect all references first
            references = []
            seen_sources = set()  # To avoid duplicates
            for i, doc in enumerate(all_documents, 1):
                # Extract metadata
                metadata = doc.metadata if hasattr(doc, "metadata") else {}
                source = metadata.get("source", "Unknown Source")
                collection = metadata.get("collection", "Unknown Collection")
                
                # Skip duplicates (optional)
                if source in seen_sources:
                    continue
                seen_sources.add(source)
                
                # Format as Obsidian URI link
                # Remove file extension if present for cleaner display
                display_name = source
                if "." in display_name:
                    display_name = display_name.rsplit(".", 1)[0]
                
                # Create Obsidian URI format obsidian://open?vault=VAULT_NAME&file=FILE_PATH
                # URL encode the file path to handle special characters
                encoded_file = urllib.parse.quote(source)
                vault_name = self.valves.OBSIDIAN_VAULT_NAME
                obsidian_uri = f"obsidian://open?vault={vault_name}&file={encoded_file}"
                
                # Create markdown link with the URI and add collection info
                references.append(f"[{i}]({obsidian_uri}) [{collection}]")
            
            # Output all references in a single line with separators
            yield " | ".join(references)
            yield "\n"