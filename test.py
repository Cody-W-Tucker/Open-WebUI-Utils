from pipelines.agent import Pipeline
import asyncio
import sys
from langchain_core.documents import Document
from typing import List, Dict, Any


# Create a mock pipeline class that extends the original Pipeline
class MockPipeline(Pipeline):
    def __init__(self):
        super().__init__()
        self.name = "Mock Context-Rich Exploratory RAG"
        
    async def on_startup(self):
        print("Starting mock pipeline initialization...")
        # Create mock document collections
        self.create_mock_collections()
        print("Mock pipeline initialized successfully!")
        
    def create_mock_collections(self):
        """Create mock document collections for testing without external services"""
        # Create mock retrievers that return predefined documents
        self.retrievers = {
            "personal": self.create_mock_retriever([
                Document(
                    page_content="Cody Tucker is an entrepreneur, marketer, and consultant known for his humanistic approach to digital marketing. He specializes in knowledge management systems and personal development frameworks.",
                    metadata={"source": "Bio.md", "collection": "personal"}
                ),
                Document(
                    page_content="In his spare time, Cody enjoys hiking, writing, and developing open-source software tools for knowledge management.",
                    metadata={"source": "Interests.md", "collection": "personal"}
                ),
            ]),
            "chat_history": self.create_mock_retriever([
                Document(
                    page_content="Who is Cody Tucker",
                    metadata={"source": "chat_2023-05-01.md", "collection": "chat_history"}
                ),
                Document(
                    page_content="Cody is the creator of the Open-WebUI-Utils project, which provides utilities for working with AI-powered web interfaces.",
                    metadata={"source": "chat_2023-05-01.md", "collection": "chat_history"}
                ),
            ]),
            "research": self.create_mock_retriever([
                Document(
                    page_content="# CODE Framework\n\nThe CODE Framework, developed by Tiago Forte, represents a systematic approach to managing information.",
                    metadata={"source": "CODE-Framework.md", "collection": "research"}
                ),
                Document(
                    page_content="# Contextual Object Theory\n\n## Origin\n\n**Immediate Purpose**: COT was created to address limitations in existing knowledge systems.",
                    metadata={"source": "COT.md", "collection": "research"}
                ),
            ]),
        }
        
        # Set up the collections that exist
        self.vector_stores = {collection: None for collection in self.retrievers.keys()}
        
    def create_mock_retriever(self, documents: List[Document]):
        """Create a mock retriever that returns predefined documents"""
        class MockRetriever:
            def __init__(self, docs):
                self.docs = docs
                
            def get_relevant_documents(self, query: str) -> List[Document]:
                # Simple filtering based on query terms
                if "cody" in query.lower() or "tucker" in query.lower():
                    return [doc for doc in self.docs if "cody" in doc.page_content.lower() or "tucker" in doc.page_content.lower()]
                return self.docs
                
            def invoke(self, input_dict: Dict[str, Any]) -> List[Document]:
                # For history-aware retriever simulation
                query = input_dict.get("input", "")
                return self.get_relevant_documents(query)
        
        return MockRetriever(documents)


async def main():
    try:
        # Use the mock pipeline instead of the real one
        pipeline = MockPipeline()
        print("Initializing pipeline...")
        await pipeline.on_startup()
        print("Pipeline initialized successfully!")
        
        # Create a list of message dictionaries to represent chat history
        messages = []  # Empty for first message, would contain previous exchanges for ongoing conversation
        
        user_query = "Who is Cody Tucker"
        print(f"\nProcessing query: '{user_query}'")
        
        # Call the pipeline with the required parameters
        response_generator = pipeline.pipe(
            user_message=user_query,
            model_id="gpt-4o-mini",
            messages=messages,
            body={}  # Empty dictionary for any additional parameters
        )
        
        # Print the streamed response
        print("\nResponse:")
        full_response = ""
        try:
            for chunk in response_generator:
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n")
        except Exception as e:
            print(f"\nError during response generation: {str(e)}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    finally:
        # Make sure we always shut down properly
        if 'pipeline' in locals():
            print("Shutting down pipeline...")
            await pipeline.on_shutdown()
            print("Pipeline shut down successfully!")
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
