# AI Travel Planning Agent v2.0

An intelligent travel planning system based on **mature six-layer AI Agent architecture**, featuring intelligent reasoning, memory learning, tool orchestration, and self-improvement capabilities.

## 🚀 Architecture Overview

This project has been completely refactored from a simple layered architecture to a mature AI Agent system with the following capabilities:

- **🧠 Intelligent Reasoning**: ReAct pattern (Reasoning + Action) with decision path tracking
- **💾 Memory Learning**: Multi-turn conversation context and user preference learning  
- **🔧 Tool Orchestration**: Dynamic tool selection, combination, and concurrent execution
- **📈 Self-Improvement**: Quality evaluation, feedback learning, and continuous optimization
- **📊 Full Monitoring**: Real-time performance monitoring and quality assessment
- **🔒 Enterprise Security**: Access control, rate limiting, and data privacy protection

## 📋 Core Features

### New AI Agent Capabilities
- **Multi-turn Context Management**: Intelligent conversation history tracking
- **Intelligent Tool Selection**: Automatic tool selection based on user requests
- **Dynamic Plan Generation**: Complex task decomposition and sequential execution
- **Conflict Detection**: Automatic identification of conflicts and issues in plans
- **User Preference Learning**: Learning from user feedback and historical behavior
- **Self-Assessment**: Automatic evaluation of output quality and continuous improvement

### Travel Planning Features
- **Personalized Itinerary Generation**: AI-powered travel planning based on user preferences
- **Real-time Integration**: Live data from flight, hotel, and attraction services
- **Multi-tool Coordination**: Intelligent orchestration of search and booking tools
- **RAG Knowledge Enhancement**: Retrieval-augmented generation to reduce hallucinations
- **Interactive Plan Refinement**: Real-time plan adjustments based on user feedback

## 🏗️ System Architecture

### Six-Layer AI Agent Architecture

```
├── ⑥ MLOps & Security Layer     (Configuration, Auth, Security)
├── ⑤ Monitoring & Evaluation    (Metrics, Quality Assessment)  
├── ④ Agent Planning & Reasoning (ReAct, Multi-Agent Coordination)
├── ③ Context & Memory Layer     (Conversation, User Profiles)
├── ② Tool & Plugin Layer        (MCP Protocol, Tool Orchestration)
└── ① Core LLM & RAG Layer       (Knowledge Retrieval, LLM Services)
```

### Project Structure

```
Project/
├── app/
│   ├── core/                   # ① Core LLM & RAG Knowledge Layer
│   │   ├── llm_service.py     # Unified LLM service interface
│   │   ├── rag_engine.py      # Knowledge retrieval engine
│   │   └── knowledge_base.py  # Travel knowledge management
│   ├── tools/                  # ② Tool & Plugin Layer  
│   │   ├── base_tool.py       # Tool base class and registry
│   │   ├── tool_executor.py   # Intelligent tool executor
│   │   ├── flight_search.py   # Flight search tool
│   │   ├── hotel_search.py    # Hotel search tool
│   │   └── attraction_search.py # Attraction search tool
│   ├── memory/                 # ③ Context & Memory Layer
│   │   ├── conversation_memory.py # Conversation memory manager
│   │   ├── user_profile.py    # User preference profiles
│   │   └── session_manager.py # Session state management
│   ├── agents/                 # ④ Agent Planning & Reasoning
│   │   ├── base_agent.py      # Agent base class and manager
│   │   ├── travel_agent.py    # Main travel planning agent
│   │   └── tool_agent.py      # Tool coordination agent
│   ├── monitoring/             # ⑤ Monitoring & Evaluation
│   │   ├── metrics.py         # Performance metrics collection
│   │   ├── evaluator.py       # Quality evaluation system
│   │   └── logger.py          # Intelligent logging system
│   ├── config/                 # ⑥ Configuration & Security
│   │   ├── settings.py        # Unified configuration management
│   │   ├── security.py        # Security mechanisms
│   │   └── auth.py            # Authentication and authorization
│   ├── knowledge/              # Knowledge Base
│   │   ├── documents/         # Travel knowledge documents
│   │   ├── schemas/           # Knowledge schemas
│   │   └── categories.yaml    # Knowledge categorization
│   ├── api/                    # API Layer
│   └── main.py                # Main application entry
├── tests/                      # Comprehensive test suite
├── ARCHITECTURE.md            # Detailed architecture documentation
└── requirements.txt           # Dependencies with AI/ML packages
```

## 🛠️ Technology Stack

### Free/Open Source Components
| Component | Technology | Reason |
|-----------|------------|---------|
| Web Framework | FastAPI | High performance, auto-documentation |
| LLM Service | OpenAI API | Best performance, free tier available |
| Vector Database | ChromaDB | Completely free and open source |
| Embedding Model | sentence-transformers | Local execution, no API costs |
| Database | SQLite/PostgreSQL | Free and open source |
| Caching | Redis | Free local deployment |
| Monitoring | OpenTelemetry + Loguru | Open source monitoring stack |
| Testing | pytest + pytest-asyncio | Comprehensive testing ecosystem |

### Optional Enterprise Components
- **Agent Framework**: LangChain Community (optional)
- **Distributed Tracing**: OpenTelemetry
- **Cloud Database**: PostgreSQL (Railway/Render free tier)
- **Cache Service**: Redis (Railway free tier)

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux  
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create `.env` file in project root:

```env
# LLM Service
OPENAI_API_KEY=your_openai_api_key_here

# Vector Database (ChromaDB - no config needed for local)
CHROMA_DB_PATH=./data/chroma_db

# Travel APIs
FLIGHT_SEARCH_API_KEY=your_flight_api_key_here
HOTEL_SEARCH_API_KEY=your_hotel_api_key_here
ATTRACTION_SEARCH_API_KEY=your_attraction_api_key_here

# Database
DATABASE_URL=sqlite:///./travel_agent.db

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True
LOG_LEVEL=INFO

# Security
SECRET_KEY=your_secure_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Agent Configuration
MAX_CONVERSATION_TURNS=50
MAX_TOOL_EXECUTION_TIME=30
ENABLE_SELF_IMPROVEMENT=true
```

### 3. Launch the System

```bash
# Start the development server
uvicorn app.main:app --reload

# The system will automatically:
# 1. Initialize the knowledge base
# 2. Register all tools  
# 3. Start AI agents
# 4. Activate memory system
```

### 4. Access the Application

- **Main Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc  
- **System Status**: http://localhost:8000/system/status

## 🧪 Testing & Validation

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test categories
pytest tests/test_rag_accuracy.py -v
```

### Validation Metrics

| Component | Metric | Target |
|-----------|--------|---------|
| RAG System | Retrieval Accuracy | >80% |
| RAG System | Response Time | <2s |
| Memory System | Conversation Continuity | >80% |
| Memory System | Preference Learning Rate | >85% |
| Reasoning System | Self-Improvement Rate | >30% |
| Reasoning System | Conflict Detection Rate | >95% |
| Monitoring System | Coverage Rate | >95% |
| Monitoring System | Anomaly Detection Rate | >90% |

## 📈 System Capabilities

### Intelligent Capabilities
- **Chain-of-Thought Reasoning**: Agents show their thinking process
- **Multi-step Planning**: Complex task decomposition and sequential execution
- **Conflict Detection**: Automatic identification of plan conflicts and issues
- **Decision Optimization**: Constraint-based solution optimization

### Memory & Learning Capabilities  
- **Short-term Memory**: Current conversation context management
- **Long-term Memory**: User preferences and historical behavior
- **Working Memory**: Current task state and intermediate results
- **Adaptive Learning**: Continuous learning from user feedback

### Tool Orchestration Capabilities
- **Intelligent Selection**: Automatic tool selection based on tasks
- **Dynamic Combination**: Multi-tool coordination and result integration
- **Concurrent Execution**: Support for parallel tool invocation
- **Error Recovery**: Intelligent retry and fallback strategies

### Self-Improvement Capabilities
- **Quality Assessment**: Automatic evaluation of output quality
- **Feedback Learning**: Learning and improvement from errors
- **Performance Optimization**: Continuous optimization of response speed and accuracy
- **Knowledge Updates**: Automatic knowledge base updates and expansion

## 🐛 Troubleshooting

### Common Issues

1. **Import/Module Errors**:
   ```bash
   # Ensure virtual environment is activated
   # Restart your IDE
   # Verify all dependencies are installed
   pip install -r requirements.txt
   ```

2. **ChromaDB Issues**:
   ```bash
   # Clear ChromaDB data if corrupted
   rm -rf ./data/chroma_db
   # Restart the application to reinitialize
   ```

3. **OpenAI API Errors**:
   ```bash
   # Verify API key is set correctly
   # Check API rate limits and quotas
   # Ensure sufficient API credits
   ```

4. **Performance Issues**:
   ```bash
   # Check system resource usage
   # Adjust MAX_CONVERSATION_TURNS in .env
   # Enable/disable self-improvement features
   ```

## 📊 Monitoring & Observability

The system includes comprehensive monitoring:

- **Real-time Metrics**: Performance indicators and system health
- **Quality Assessment**: Automatic evaluation of responses and user satisfaction  
- **Error Tracking**: Intelligent error detection and recovery
- **Usage Analytics**: Tool usage patterns and optimization recommendations

Access monitoring dashboard at: http://localhost:8000/system/status

## 🔒 Security & Privacy

- **API Key Security**: Secure storage and rotation of API keys
- **Access Control**: User authentication and authorization
- **Rate Limiting**: Protection against abuse and overuse
- **Data Privacy**: User data protection and consent management
- **Audit Logging**: Comprehensive activity tracking

## 📚 Documentation

- **Architecture Guide**: See `ARCHITECTURE.md` for detailed system design
- **API Reference**: Auto-generated at http://localhost:8000/docs
- **Development Guide**: Check `/docs` folder for development guidelines

## 🤝 Contributing

This is a course project implementing mature AI Agent architecture patterns. The system demonstrates:

- Industry-standard six-layer AI Agent design
- Integration of RAG, memory, and reasoning systems
- Tool orchestration and self-improvement capabilities
- Enterprise-grade monitoring and security

Perfect for learning modern AI Agent development and deployment patterns.

## 📄 License

MIT License - See LICENSE file for details 