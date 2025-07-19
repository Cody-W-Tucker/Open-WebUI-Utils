"""
title: Journal RAG
author: Cody W Tucker
date: 2024-12-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using langchain with citations.
requirements: langchain==0.3.3, langchain_core==0.3.10, langchain_openai==0.3.18, openai==1.82.0, langchain_qdrant==0.2.0, qdrant_client==1.11.0, pydantic==2.7.4, langchain_ollama
"""

import os
from typing import List, Union, Generator, Iterator
import pydantic
import sys
print(f"Loaded Pydantic version: {pydantic.__version__}")
print(f"Pydantic module path: {pydantic.__file__}")

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document

class Pipeline:
    class Valves(BaseModel):
        OPENAI_API_KEY: str
        OPENAI_MODEL: str
        QDRANT_URL: str
        SYSTEM_PROMPT: str
        OBSIDIAN_VAULT_NAME: str
        OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text:latest"
        OLLAMA_BASE_URL: str = "http://localhost:11434"
        model_config = {"extra": "allow"}

    def __init__(self):
        self.name = "Journal RAG"
        self.vector_store = None
        self.llm = None
        self.valves = self.Valves(**{
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "your-api-key-here"),
            "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "QDRANT_URL": os.getenv("QDRANT_URL", "http://localhost:6333"),
            "SYSTEM_PROMPT": os.getenv("SYSTEM_PROMPT", 
                """You are an assistant for question-answering tasks. 
                Use the following pieces of retrieved context to answer 
                the question. If you don't know the answer, say that you 
                don't know.
                """),
            "OBSIDIAN_VAULT_NAME": os.getenv("OBSIDIAN_VAULT_NAME", "MyVault"),
            "OLLAMA_EMBEDDING_MODEL": os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest"),
            "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        })

    async def on_startup(self):
        from langchain_openai import ChatOpenAI
        from langchain_ollama import OllamaEmbeddings
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        
        try:
            print("Starting initialization...")
            client = QdrantClient(url=self.valves.QDRANT_URL, https=True)
            print("Qdrant client initialized")
            
            if not client.collection_exists("personal"):
                raise ValueError("Collection 'personal' does not exist")
            print("Collection exists")
            
            embeddings = OllamaEmbeddings(
                model=self.valves.OLLAMA_EMBEDDING_MODEL,
                base_url=self.valves.OLLAMA_BASE_URL,
            )
            print(f"Embeddings initialized with Ollama model {self.valves.OLLAMA_EMBEDDING_MODEL} at {self.valves.OLLAMA_BASE_URL}")
            
            self.vector_store = QdrantVectorStore(
                client=client,
                collection_name="personal",
                embedding=embeddings
            )
            print("Vector store initialized")
            
            self.llm = ChatOpenAI(
                model=self.valves.OPENAI_MODEL,
                api_key=self.valves.OPENAI_API_KEY,
                streaming=True,
                temperature=1
            )
            print("LLM initialized")
            
        except Exception as e:
            print(f"Failed to initialize: {str(e)}")
            raise

    async def on_shutdown(self):
        pass

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Generator[str, None, None]:
        retriever = self.vector_store.as_retriever()
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question, "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is."
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
        )

        if not self.llm or not retriever:
            raise ValueError("LLM or retriever not properly initialized")

        history_aware_retriever = create_history_aware_retriever(self.llm, retriever, contextualize_q_prompt)
        
        # Always append the context tag to the system prompt
        system_prompt = self.valves.SYSTEM_PROMPT

        # Define footnote additions
        footnote_additions = """
        IMPORTANT: You MUST include footnote references [^n] in your response whenever you use information from the retrieved context.
        
        Rules for using footnotes:
        1. Add [^n] at the end of any sentence that uses information from the context
        2. Use [^n] after specific facts, quotes, or ideas from the sources
        3. You can use multiple footnotes for a single sentence if it combines information from different sources
        4. Number the footnotes sequentially starting from [^1]
        5. Make sure every piece of information from the context has a corresponding footnote
        6. DO NOT add any descriptive text after the footnote numbers - just use [^n] format
        7. DO NOT create your own footnote section - the system will automatically add the References section
        
        Example response:
        "The project was first announced in 2020 [^1] and has since grown significantly [^2]. According to the latest report [^3], it now serves over 1 million users. The team's innovative approach [^4] has been key to this success."

        The system will automatically add the References section with the correct links at the end of your response. NEVER include a References section in your response.
        """

        # Define context format
        context_formatted = """
        Context to pull references from:

        {context}
        """

        # Clean up any trailing whitespace and add the context tag
        system_prompt = system_prompt.rstrip() + footnote_additions + context_formatted

        qa_prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        response_stream = rag_chain.stream({"chat_history": messages, "input": user_message})
        
        # Collect documents for citations
        documents = []
        has_answer = False

        # Stream the answer chunks
        for chunk in response_stream:
            if isinstance(chunk, dict):
                if "context" in chunk:
                    # Store retrieved documents
                    documents = chunk["context"] if isinstance(chunk["context"], list) else []
                if "answer" in chunk:
                    has_answer = True
                    yield chunk["answer"]

        # If no answer was streamed, yield a fallback
        if not has_answer:
            yield "I don't know."

        # Append citations if documents were retrieved
        if documents:
            # Check if we already have a references section
            has_references = False
            for chunk in response_stream:
                if isinstance(chunk, str) and "References:" in chunk:
                    has_references = True
                    break
            
            # Only add a references section if one doesn't already exist
            if not has_references:
                # Collect all references first
                references = []
                seen_sources = set()  # To avoid duplicates
                for i, doc in enumerate(documents, 1):
                    # Extract metadata
                    metadata = doc.metadata if isinstance(doc, Document) and hasattr(doc, "metadata") else {}
                    source = metadata.get("source", "Unknown Source")

                    # Skip duplicates
                    if source in seen_sources:
                        continue
                    seen_sources.add(source)
                    
                    # Remove file extension if present for cleaner display
                    display_name = source
                    if "." in display_name:
                        display_name = display_name.rsplit(".", 1)[0]
                    
                    # Create numbered footnote format
                    references.append(f"[^{i}]: [[{display_name}]]")
                
                # Output references as numbered footnotes
                yield "\n".join(references)
                yield "\n"