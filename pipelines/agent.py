"""
title: Context-Rich Exploratory RAG Chatbot
author: Cody W Tucker
date: 2024-12-30
version: 1.0
license: MIT
description: A pipeline for retrieving and synthesizing information across multiple knowledge bases with relationship detection, contradiction analysis, and follow-up suggestions.
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

class Pipeline:
    class Valves(BaseModel):
        OPENAI_API_KEY: str
        OPENAI_MODEL: str = "gpt-4o-mini"
        OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"
        QDRANT_URLS: List[str] = Field(default_factory=list)
        QDRANT_COLLECTIONS: List[str] = Field(default_factory=list)
        COLLECTION_DESCRIPTIONS: Dict[str, str] = Field(default_factory=dict)
        SYSTEM_PROMPT: str
        OBSIDIAN_VAULT_NAME: str
        MAX_DOCUMENTS_PER_COLLECTION: int = 5
        TEMPERATURE: float = 0.7
        FOLLOW_UP_ENABLED: bool = True
        CONTRADICTION_DETECTION_ENABLED: bool = True
        RELATIONSHIP_ANALYSIS_ENABLED: bool = True
        ENSEMBLE_WEIGHTS: Dict[str, float] = Field(default_factory=dict)
        model_config = {"extra": "allow"}

    def __init__(self):
        self.name = "Context-Rich Exploratory RAG"
        self.vector_stores = {}  # Dict mapping collection name to vector store
        self.retrievers = {}  # Dict mapping collection name to retriever
        self.llm = None
        self.embeddings = None
        
        # Parse environment variables
        qdrant_urls = os.getenv("QDRANT_URLS", "http://qdrant.homehub.tv").split(",")
        qdrant_collections = os.getenv("QDRANT_COLLECTIONS", "personal").split(",")
        
        # Parse collection descriptions if available
        collection_descriptions = {}
        desc_str = os.getenv("COLLECTION_DESCRIPTIONS", "")
        if desc_str:
            pairs = desc_str.split(";")
            for pair in pairs:
                if ":" in pair:
                    name, desc = pair.split(":", 1)
                    collection_descriptions[name.strip()] = desc.strip()
                    
        # Set default descriptions if not provided
        for collection in qdrant_collections:
            if collection not in collection_descriptions:
                collection_descriptions[collection] = f"{collection} knowledge base"
        
        # Parse ensemble weights if available
        ensemble_weights = {}
        weights_str = os.getenv("ENSEMBLE_WEIGHTS", "")
        if weights_str:
            pairs = weights_str.split(";")
            for pair in pairs:
                if ":" in pair:
                    name, weight = pair.split(":", 1)
                    try:
                        ensemble_weights[name.strip()] = float(weight.strip())
                    except ValueError:
                        print(f"Invalid weight value for {name}: {weight}")
        
        # Set default weights if not provided (equal weights)
        if not ensemble_weights and qdrant_collections:
            default_weight = 1.0 / len(qdrant_collections)
            for collection in qdrant_collections:
                ensemble_weights[collection] = default_weight
        
        self.valves = self.Valves(**{
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "your-api-key-here"),
            "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "OPENAI_EMBEDDING_MODEL": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
            "QDRANT_URLS": qdrant_urls,
            "QDRANT_COLLECTIONS": qdrant_collections,
            "COLLECTION_DESCRIPTIONS": collection_descriptions,
            "ENSEMBLE_WEIGHTS": ensemble_weights,
            "SYSTEM_PROMPT": os.getenv("SYSTEM_PROMPT", 
                """You are a dynamic knowledge partner capable of synthesizing insights across multiple sources.
                Use the provided context to not just answer questions, but to highlight connections, 
                identify patterns, and suggest new perspectives. When appropriate, note contradictions
                or tensions between different sources. Push the user's thinking by suggesting follow-up
                directions or unexplored angles. If you don't know the answer, acknowledge that
                and suggest alternative approaches."""),
            "OBSIDIAN_VAULT_NAME": os.getenv("OBSIDIAN_VAULT_NAME", "MyVault"),
            "MAX_DOCUMENTS_PER_COLLECTION": int(os.getenv("MAX_DOCUMENTS_PER_COLLECTION", "5")),
            "TEMPERATURE": float(os.getenv("TEMPERATURE", "0.7")),
            "FOLLOW_UP_ENABLED": os.getenv("FOLLOW_UP_ENABLED", "true").lower() == "true",
            "CONTRADICTION_DETECTION_ENABLED": os.getenv("CONTRADICTION_DETECTION_ENABLED", "true").lower() == "true",
            "RELATIONSHIP_ANALYSIS_ENABLED": os.getenv("RELATIONSHIP_ANALYSIS_ENABLED", "true").lower() == "true",
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
            print(f"Attempting to connect to collections: {', '.join(self.valves.QDRANT_COLLECTIONS)}")
            print(f"Using Qdrant URLs: {', '.join(self.valves.QDRANT_URLS)}")
            
            # Create a client for each URL
            clients = {}
            for url in self.valves.QDRANT_URLS:
                try:
                    clients[url] = QdrantClient(url=url, https=url.startswith("https"))
                    print(f"Connected to Qdrant at {url}")
                except Exception as e:
                    print(f"Failed to connect to Qdrant at {url}: {str(e)}")
            
            if not clients:
                raise ValueError("Could not connect to any Qdrant servers")
            
            # Initialize each vector store
            # Check each collection on each client
            for collection in self.valves.QDRANT_COLLECTIONS:
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
                            retriever = vector_store.as_retriever(
                                search_kwargs={"k": self.valves.MAX_DOCUMENTS_PER_COLLECTION}
                            )
                            self.retrievers[collection] = retriever
                            
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
                temperature=self.valves.TEMPERATURE
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

    def _analyze_document_relationships(self, documents: List[Document]) -> Dict[str, Any]:
        """Analyze relationships between retrieved documents."""
        if not documents or not self.valves.RELATIONSHIP_ANALYSIS_ENABLED:
            return {}
            
        # Group documents by collection
        collections = {}
        for doc in documents:
            collection = doc.metadata.get("collection", "unknown")
            if collection not in collections:
                collections[collection] = []
            collections[collection].append(doc)
            
        # Look for chronological patterns
        dates = []
        for doc in documents:
            date_str = doc.metadata.get("date")
            if date_str:
                try:
                    # Try to parse date in various formats
                    for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%B %d, %Y"]:
                        try:
                            date = datetime.strptime(date_str, fmt)
                            dates.append((date, doc))
                            break
                        except ValueError:
                            continue
                except Exception:
                    pass
                    
        # Sort by date if we have dates
        chronological_insights = None
        if len(dates) > 1:
            dates.sort(key=lambda x: x[0])
            chronological_insights = {
                "earliest": dates[0][1].page_content[:100] + "...",
                "latest": dates[-1][1].page_content[:100] + "...",
                "span_days": (dates[-1][0] - dates[0][0]).days
            }
            
        # Look for thematic overlap using simple keyword matching
        # In a more advanced implementation, we could use embeddings or LLM for this
        keywords = {}
        for doc in documents:
            words = doc.page_content.lower().split()
            for word in words:
                if len(word) > 4:  # Only consider words longer than 4 characters
                    if word not in keywords:
                        keywords[word] = 0
                    keywords[word] += 1
                    
        # Find common themes (words that appear in multiple documents)
        common_themes = [word for word, count in keywords.items() if count > 1]
        
        return {
            "collections_represented": list(collections.keys()),
            "chronological_insights": chronological_insights,
            "common_themes": common_themes[:5],  # Top 5 common themes
            "cross_collection": len(collections) > 1
        }

    def _detect_contradictions(self, documents: List[Document], query: str) -> List[Dict[str, Any]]:
        """Detect contradictions between documents."""
        if not documents or len(documents) < 2 or not self.valves.CONTRADICTION_DETECTION_ENABLED:
            return []
            
        # Simple implementation - just a placeholder
        # In a real implementation, we would use the LLM to analyze documents for contradictions
        # This would be done by creating pairs of documents and asking the LLM if they contradict
        
        # For now, return an empty list
        return []

    def _generate_follow_up_questions(self, documents: List[Document], query: str, answer: str) -> List[str]:
        """Generate follow-up questions based on retrieved documents and the answer."""
        if not documents or not self.valves.FOLLOW_UP_ENABLED:
            return []
            
        # Simple implementation - just a placeholder
        # In a real implementation, we would use the LLM to generate follow-up questions
        
        # For now, return an empty list
        return []

    def _create_contextualized_ensemble_retriever(self, relevant_collections: List[str], messages: List[dict]) -> Optional[EnsembleRetriever]:
        """Create an EnsembleRetriever from history-aware retrievers for the relevant collections."""
        if not relevant_collections:
            return None
            
        # Filter to only collections we have retrievers for
        available_collections = [col for col in relevant_collections if col in self.retrievers]
        if not available_collections:
            return None
            
        # Create history-aware retrievers for each collection
        history_aware_retrievers = []
        retriever_weights = []
        
        for collection in available_collections:
            base_retriever = self.retrievers[collection]
            collection_desc = self.valves.COLLECTION_DESCRIPTIONS.get(collection, f"{collection} knowledge base")
            
            # Create contextualization prompt for this collection
            contextualize_q_system_prompt = (
                f"Given a chat history and the latest user question, "
                f"formulate a standalone question which can be understood "
                f"without the chat history. This query will be used to search the '{collection}' collection, "
                f"which contains {collection_desc}. "
                f"Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
            )
            
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
            )
            
            # Create history-aware retriever for this collection
            history_aware_retriever = create_history_aware_retriever(
                self.llm, base_retriever, contextualize_q_prompt
            )
            
            # Add collection metadata to documents (via a wrapper function)
            def add_collection_metadata(input_dict):
                # Get original documents
                documents = history_aware_retriever.invoke(input_dict)
                # Add collection metadata
                for doc in documents:
                    if hasattr(doc, "metadata"):
                        doc.metadata["collection"] = collection
                    else:
                        doc.metadata = {"collection": collection}
                return documents
            
            # Create a RunnableLambda to add metadata
            retriever_with_metadata = RunnableLambda(add_collection_metadata)
            
            # Add to our lists
            history_aware_retrievers.append(retriever_with_metadata)
            
            # Get weight for this collection (default to equal weighting if not specified)
            weight = self.valves.ENSEMBLE_WEIGHTS.get(collection, 1.0 / len(available_collections))
            retriever_weights.append(weight)
            
        # Create and return the ensemble retriever
        if history_aware_retrievers:
            return EnsembleRetriever(
                retrievers=history_aware_retrievers,
                weights=retriever_weights
            )
        return None

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Generator[str, None, None]:
        # Determine which collections to query
        relevant_collections = self._determine_relevant_collections(user_message, messages)
        
        # Create an ensemble retriever with history-aware retrievers
        ensemble_retriever = self._create_contextualized_ensemble_retriever(relevant_collections, messages)
        
        # If we couldn't create an ensemble retriever, respond accordingly
        if not ensemble_retriever:
            print("Could not create ensemble retriever")
            yield "I couldn't access any knowledge bases. Please try again later or contact support."
            return
            
        # Retrieve documents using the ensemble retriever
        try:
            all_documents = ensemble_retriever.invoke({"chat_history": messages, "input": user_message})
            print(f"Retrieved {len(all_documents)} documents from ensemble retriever")
            for i, doc in enumerate(all_documents):
                collection = doc.metadata.get("collection", "unknown")
                print(f"Doc {i} from {collection}: {doc.page_content[:100]}...")
        except Exception as e:
            print(f"Error retrieving from ensemble retriever: {str(e)}")
            yield f"Error retrieving information: {str(e)}"
            return
        
        # If no documents were retrieved, respond accordingly
        if not all_documents:
            print("No documents were retrieved from any collection")
            yield "I couldn't find any relevant information in my knowledge base. Could you provide more details or try a different question?"
            return
            
        # Make sure we have content in the documents
        has_content = False
        for doc in all_documents:
            if doc.page_content and len(doc.page_content.strip()) > 0:
                has_content = True
                break
                
        if not has_content:
            print("Documents were retrieved but they have no content")
            yield "I found documents but they don't contain useful information. Please try a different query."
            return
            
        # Analyze document relationships
        relationship_analysis = self._analyze_document_relationships(all_documents)
        
        # Detect contradictions
        contradictions = self._detect_contradictions(all_documents, user_message)
        
        # Prepare system prompt with relationship and contradiction information
        system_prompt = self.valves.SYSTEM_PROMPT
        
        # Add relationship analysis to system prompt if available
        if relationship_analysis and self.valves.RELATIONSHIP_ANALYSIS_ENABLED:
            system_prompt += "\n\nRelationship Analysis:"
            if relationship_analysis.get("cross_collection"):
                system_prompt += f"\n- Information retrieved from multiple collections: {', '.join(relationship_analysis.get('collections_represented', []))}"
            if relationship_analysis.get("chronological_insights"):
                insights = relationship_analysis["chronological_insights"]
                system_prompt += f"\n- Documents span {insights.get('span_days', 'unknown')} days"
            if relationship_analysis.get("common_themes"):
                system_prompt += f"\n- Common themes across documents: {', '.join(relationship_analysis.get('common_themes', []))}"
        
        # Add contradiction information to system prompt if available
        if contradictions and self.valves.CONTRADICTION_DETECTION_ENABLED:
            system_prompt += "\n\nPotential Contradictions:"
            for i, contradiction in enumerate(contradictions, 1):
                system_prompt += f"\n{i}. {contradiction.get('description', 'Unknown contradiction')}"
        
        # Print the documents we're using
        print(f"Using {len(all_documents)} total documents for answer generation")
        for i, doc in enumerate(all_documents):
            print(f"Final doc {i}: {doc.page_content[:100]}... from {doc.metadata.get('collection', 'unknown')}")
        
        # Format documents into a context string for easier debugging
        formatted_context = "\n\n".join([f"Document {i+1} from {doc.metadata.get('collection', 'unknown')}:\n{doc.page_content}" 
                                        for i, doc in enumerate(all_documents)])
        print(f"Context length: {len(formatted_context)} characters")
        
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
            
        # Generate follow-up questions
        follow_up_questions = self._generate_follow_up_questions(all_documents, user_message, answer)
        
        # Add follow-up questions if available
        if follow_up_questions and self.valves.FOLLOW_UP_ENABLED:
            yield "\n\nFollow-up Questions:\n"
            for i, question in enumerate(follow_up_questions, 1):
                yield f"{i}. {question}\n"
        
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