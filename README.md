# Open-WebUI PocketFlow Chat Agent

A clean Open-WebUI pipeline that uses **actual PocketFlow Node and Flow abstractions** with a **hybrid approach**: **local Ollama for task routing** and **OpenAI for high-quality chat responses**.

## Overview

This pipeline demonstrates how to integrate the real PocketFlow framework with Open-WebUI using a **best-of-both-worlds approach**. PocketFlow is a minimalist LLM framework that uses **Nodes** and **Flows** to create clean, modular AI applications. This implementation uses:

- **Ollama** for collection routing (local, fast, private task model)
- **OpenAI** for main chat responses (high-quality, context-aware conversations)
- **Ollama embeddings** for vector search (local, private document retrieval)

## Key Features

### 🎯 **Real PocketFlow Integration**
- **CollectionRouterNode**: Routes queries using local Ollama (llama3.2:latest)
- **DocumentRetrieverNode**: Retrieves documents with Ollama embeddings (nomic-embed-text)
- **DocumentProcessorNode**: Extracts insights following PocketFlow architecture
- **ResponseGeneratorNode**: Generates responses using OpenAI for quality
- **Flow orchestration**: Connects nodes with PocketFlow's `>>` syntax

### 🔄 **PocketFlow Architecture**
```python
# True PocketFlow pattern
router_node - "retrieve" >> retriever_node
retriever_node - "process" >> processor_node  
processor_node - "generate" >> generator_node

chat_flow = Flow(start=router_node)
chat_flow.run(shared)
```

### ⚡ **Hybrid Stack (Best of Both Worlds)**
- **PocketFlow**: Node and Flow abstractions for clean AI workflows
- **Ollama**: Local task routing and embeddings (private, fast, free)
- **OpenAI**: High-quality chat responses (excellent conversation quality)
- **Qdrant**: Vector database for document storage
- **Open-WebUI**: Pipeline hosting and UI

## File Structure

```
pipelines/
├── pocket_flow_agent.py    # Real PocketFlow Node/Flow pipeline (hybrid)
├── agent.py               # Complex multi-retriever agent (existing)
├── obsidian_rag.py        # Simple RAG pipeline (existing)
└── test_pipeline.py       # Basic test pipeline (existing)

README.md                 # This documentation
```

## Dependencies

The pipeline balances local and cloud capabilities:
```
pocketflow      # Node/Flow abstractions
openai          # High-quality chat responses
ollama          # Local task routing & embeddings
qdrant_client   # Vector database
```

## PocketFlow Node Architecture (Hybrid)

### 1. CollectionRouterNode (Local Ollama)
```python
class CollectionRouterNode(Node):
    def prep(self, shared):
        # Extract query and available collections
        return {"query": shared["user_query"], "collections": shared["available_collections"]}
    
    def exec(self, inputs):
        # Use LOCAL llama3.2:latest for fast, private collection routing
        response = ollama.chat(
            model=shared.get("ollama_task_model", "llama3.2:latest"),
            messages=[{"role": "user", "content": routing_prompt}],
            options={"temperature": 0}  # Deterministic routing
        )
        # Returns: ["personal", "research"] 
    
    def post(self, shared, prep_res, exec_res):
        shared["selected_collections"] = exec_res
        return "retrieve"  # Next action
```

### 2. DocumentRetrieverNode (Ollama Embeddings)
```python
class DocumentRetrieverNode(Node):
    def prep(self, shared):
        # Get query and selected collections
        return {"query": shared["user_query"], "collections": shared["selected_collections"]}
    
    def exec(self, inputs):
        # Use LOCAL nomic-embed-text for private document search
        query_embedding = ollama.embeddings(model="nomic-embed-text:latest", prompt=query)
        # Search Qdrant with local embeddings
        # Returns: [Document, Document, ...]
    
    def post(self, shared, prep_res, exec_res):
        shared["documents"] = exec_res
        return "process"  # Next action
```

### 3. DocumentProcessorNode (Local Processing)
```python
class DocumentProcessorNode(Node):
    def prep(self, shared):
        return shared.get("documents", [])
    
    def exec(self, documents):
        # Extract insights and detect contradictions (local processing)
        # Returns: ["Cross-collection pattern found", "Contradiction detected"]
    
    def post(self, shared, prep_res, exec_res):
        shared["insights"] = exec_res
        return "generate"  # Next action
```

### 4. ResponseGeneratorNode (OpenAI Quality)
```python
class ResponseGeneratorNode(Node):
    def prep(self, shared):
        # Gather all context for response generation
        return {"query": shared["user_query"], "documents": shared["documents"], ...}
    
    def exec(self, inputs):
        # Build system prompt with context and insights
        # Returns: {"system_prompt": "...", "query": "...", "documents": [...]}
    
    def post(self, shared, prep_res, exec_res):
        shared["response_context"] = exec_res
        return "default"  # End flow - will use OpenAI for final response
```

## Flow Orchestration

```python
# Connect nodes with PocketFlow syntax
router_node - "retrieve" >> retriever_node      # Ollama routing
retriever_node - "process" >> processor_node    # Ollama embeddings  
processor_node - "generate" >> generator_node   # Local processing

# Create flow
chat_flow = Flow(start=router_node)

# Execute hybrid flow
shared = {"user_query": "What are my thoughts on AI?", ...}
chat_flow.run(shared)

# Final response uses OpenAI for quality
openai_response = openai.chat.completions.create(...)
```

## Installation & Setup

### 1. Install Local Models
```bash
# Install the task routing model (you already have this)
ollama pull llama3.2:latest

# Install the embedding model
ollama pull nomic-embed-text:latest
```

### 2. Environment Variables
```bash
# OpenAI for high-quality chat responses
export OPENAI_API_KEY="your-openai-api-key"

# Ollama for local tasks (optional - defaults provided)
export OLLAMA_TASK_MODEL="llama3.2:latest"
export OLLAMA_EMBEDDING_MODEL="nomic-embed-text:latest"
export OLLAMA_BASE_URL="http://localhost:11434"

# Qdrant for vector storage
export QDRANT_URL="http://localhost:6333"
```

### 3. Vector Store Collections
Ensure you have these Qdrant collections:
- `personal` - Personal notes, thoughts, and journal entries
- `research` - Research papers, articles, and academic content
- `projects` - Work projects, code, and technical documentation
- `entities` - People, places, organizations, and other entities
- `chat_history` - Previous conversations and interactions

### 4. Deploy to Open-WebUI
1. Copy `pipelines/pocket_flow_agent.py` to your Open-WebUI pipelines directory
2. The pipeline will appear as "PocketFlow Chat Agent" in your model selection
3. Configure the valves (API key, model names, URLs) through the Open-WebUI interface
4. Start chatting with hybrid AI!

## Configuration (Valves)

- `OPENAI_API_KEY`: Your OpenAI API key (for chat responses)
- `OPENAI_MODEL`: OpenAI model to use (default: "gpt-4o-mini-2024-07-18")
- `OLLAMA_TASK_MODEL`: Local model for routing (default: "llama3.2:latest")
- `OLLAMA_EMBEDDING_MODEL`: Local embedding model (default: "nomic-embed-text:latest")
- `OLLAMA_BASE_URL`: Ollama server URL (default: "http://localhost:11434")
- `QDRANT_URL`: Qdrant vector database URL
- `MAX_DOCS_PER_COLLECTION`: Maximum documents to retrieve per collection
- `ENABLE_INSIGHTS`: Enable/disable insight extraction

## Hybrid Approach Benefits

### 🏠 **Local Tasks (Private & Fast)**
- **Collection routing**: Private, no API costs
- **Document embeddings**: Local vector search
- **Insight processing**: No external calls
- **Fast task completion**: No network latency

### ☁️ **Cloud Quality (Best Responses)**
- **Chat responses**: OpenAI's superior conversation quality
- **Context understanding**: Better synthesis of complex information
- **Nuanced responses**: High-quality reasoning and creativity
- **Reliable streaming**: Robust response generation

### 💡 **Best of Both Worlds**
- **Privacy**: Sensitive routing and search stays local
- **Quality**: Final responses use best-in-class models
- **Cost efficiency**: Only pay for final response generation
- **Speed**: Local processing for fast tasks, cloud for quality

## Example Flow Execution

```
User Query: "What are my thoughts on AI research?"

1. CollectionRouterNode (LOCAL)
   - prep(): Extract query from shared context
   - exec(): Use llama3.2:latest locally to determine ["personal", "research"]
   - post(): Store selected collections, return "retrieve"

2. DocumentRetrieverNode (LOCAL)
   - prep(): Get query and selected collections
   - exec(): Use nomic-embed-text locally, search Qdrant, retrieve 6 documents
   - post(): Store documents, return "process"

3. DocumentProcessorNode (LOCAL)
   - prep(): Get documents from shared context
   - exec(): Extract insights locally: ["Cross-collection pattern found"]
   - post(): Store insights, return "generate"

4. ResponseGeneratorNode (PREPARE FOR CLOUD)
   - prep(): Gather all context (query, documents, insights)
   - exec(): Build comprehensive system prompt with context
   - post(): Store response context, return "default" (end)

5. Pipeline.pipe() (OPENAI QUALITY)
   - Use response context to stream high-quality OpenAI response
   - Emit citations for source documents
```

## Comparison: Hybrid vs Pure Approaches

| Aspect | Pure Local (Ollama) | Pure Cloud (OpenAI) | Hybrid (This Pipeline) |
|--------|---------------------|---------------------|------------------------|
| **Privacy** | Full privacy | Data sent to cloud | Routing/search private, responses cloud |
| **Cost** | Free after setup | Pay per token | Minimal costs (routing free) |
| **Speed** | Fast local | Network latency | Fast routing, quality responses |
| **Quality** | Good for tasks | Excellent responses | Best tasks + best responses |
| **Offline** | Works offline | Requires internet | Partial offline capability |
| **Setup** | Local models only | API key only | Both (but optimized) |

## Why This Hybrid Approach?

### 🔒 **Privacy Where It Matters**
- Collection routing decisions stay local
- Document search and embeddings never leave your machine
- Only final response context goes to OpenAI

### 💰 **Cost Optimization**
- Expensive operations (routing, embeddings) run locally
- Only pay for high-quality final response generation
- Typical cost: <$0.01 per conversation vs >$0.10 for pure OpenAI

### ⚡ **Performance Optimization**
- Fast local decisions (routing, search)
- High-quality final responses (OpenAI)
- Best latency profile for user experience

### 🛠 **Maintenance Simplicity**
- Local models handle simple, deterministic tasks
- OpenAI handles complex reasoning and conversation
- Each component does what it's best at

## Contributing

When extending this pipeline:

1. **Follow PocketFlow patterns**: Use `prep()`, `exec()`, `post()` in nodes
2. **Use shared dictionary**: Pass data through the shared context
3. **Connect with actions**: Use `node - "action" >> next_node` syntax
4. **Keep nodes focused**: Each node should have a single, clear purpose
5. **Consider hybrid opportunities**: Use local for simple tasks, cloud for quality

## Troubleshooting

### Local Models
```bash
# Check Ollama models
ollama list
ollama pull llama3.2:latest
ollama pull nomic-embed-text:latest

# Test Ollama
curl http://localhost:11434/api/tags
```

### OpenAI Setup
```bash
# Test API key
export OPENAI_API_KEY="your-key"
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models
```

### Qdrant Issues
- Verify Qdrant is running and accessible
- Check collection names and schema
- Ensure embeddings dimension matches

## License

MIT License - see the pipeline header for full details.

---

*This pipeline demonstrates how PocketFlow's 100-line philosophy can orchestrate hybrid AI systems that balance privacy, cost, and quality by using the right tool for each task.* 