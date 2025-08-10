# Open Deep Research

<img width="1388" height="298" alt="full_diagram" src="https://github.com/user-attachments/assets/12a2371b-8be2-4219-9b48-90503eb43c69" />

Deep research has broken out as one of the most popular agent applications. This is a comprehensive, configurable, fully open source deep research agent that works across many model providers, search tools, and MCP servers with integrated payment processing, subscription management, and advanced RAG capabilities.

**🆕 New Features:**
- **Subscription-Based Access Control**: Free, Pro ($29.99/month), and Enterprise ($99.99/month) tiers
- **Payment Processing**: Integrated Stripe payment system with webhook handling
- **Advanced RAG System**: Vector-based document storage and retrieval for enhanced research
- **Usage Tracking**: Comprehensive API usage monitoring and rate limiting
- **MCP Server Integration**: Reddit, YouTube, and GitHub (Pro/Enterprise) MCP servers
- **FastAPI Backend**: Production-ready API with authentication and subscription management

* Read more in our [blog](https://blog.langchain.com/open-deep-research/) 
* See our [video](https://www.youtube.com/watch?v=agGiWUpxkhg) for a quick overview

### 🚀 Quickstart

1. Clone the repository and activate a virtual environment:
```bash
git clone https://github.com/langchain-ai/open_deep_research.git
cd open_deep_research
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
uv pip install -r pyproject.toml
```

3. Set up your `.env` file to customize the environment variables (for model selection, search tools, and other configuration settings):
```bash
cp .env.example .env
```

4. Launch the assistant with the LangGraph server locally to open LangGraph Studio in your browser:

```bash
# Install dependencies and start the LangGraph server
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
```

Use this to open the Studio UI:
```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```
<img width="817" height="666" alt="Screenshot 2025-07-13 at 11 21 12 PM" src="https://github.com/user-attachments/assets/052f2ed3-c664-4a4f-8ec2-074349dcaa3f" />

Ask a question in the `messages` input field and click `Submit`.

---

## 🔧 Complete Setup Guide

### Prerequisites

- Python 3.11+
- Node.js 18+ (for MCP servers)
- PostgreSQL database (via Supabase)
- Stripe account (for payments)
- GitHub API token (for GitHub MCP server)

### Production Setup with Full Features

For production deployment with payment processing, subscription management, and advanced features:

#### 1. Environment Setup

1. Clone and setup the repository:
```bash
git clone https://github.com/yasir23/Qunata_AIbackend.git
cd Qunata_AIbackend
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

2. Copy and configure environment variables:
```bash
cp .env.example .env
```

3. Edit `.env` with your configuration:
```bash
# Core API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...

# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# Stripe Configuration
STRIPE_PUBLISHABLE_KEY=pk_test_...
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_PRO_PRICE_ID=price_...
STRIPE_ENTERPRISE_PRICE_ID=price_...

# GitHub API (for GitHub MCP server)
GITHUB_TOKEN=ghp_...

# Optional: LangSmith (for tracing)
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=open-deep-research
```

---

## 💳 Payment System Setup

### Stripe Configuration

1. **Create a Stripe Account**
   - Sign up at [stripe.com](https://stripe.com)
   - Complete account verification

2. **Get API Keys**
   - Go to Developers → API keys
   - Copy your Publishable key and Secret key
   - Add them to your `.env` file

3. **Create Products and Prices**
   ```bash
   # Create Pro subscription product
   stripe products create \
     --name "Pro Plan" \
     --description "Advanced research features with GitHub MCP access"

   # Create Pro price (monthly)
   stripe prices create \
     --product prod_... \
     --unit-amount 2999 \
     --currency usd \
     --recurring interval=month

   # Create Enterprise subscription product
   stripe products create \
     --name "Enterprise Plan" \
     --description "Unlimited research with dedicated support"

   # Create Enterprise price (monthly)
   stripe prices create \
     --product prod_... \
     --unit-amount 9999 \
     --currency usd \
     --recurring interval=month
   ```

4. **Setup Webhooks**
   - Go to Developers → Webhooks
   - Add endpoint: `https://your-domain.com/webhooks/stripe`
   - Select events: `customer.subscription.created`, `customer.subscription.updated`, `customer.subscription.deleted`, `invoice.payment_succeeded`, `invoice.payment_failed`
   - Copy the webhook secret to your `.env` file

5. **Test Webhook Locally** (for development):
   ```bash
   # Install Stripe CLI
   stripe listen --forward-to localhost:8000/webhooks/stripe
   ```

---

## 🗄️ Database Setup

### Supabase Configuration

1. **Create Supabase Project**
   - Sign up at [supabase.com](https://supabase.com)
   - Create a new project
   - Note your project URL and anon key

2. **Run Database Migrations**
   ```sql
   -- Run these SQL commands in Supabase SQL Editor
   
   -- Enable pgvector extension for RAG
   CREATE EXTENSION IF NOT EXISTS vector;
   
   -- Run migration files in order:
   -- 1. src/database/migrations/001_create_payment_tables.sql
   -- 2. src/database/migrations/002_create_vector_store_tables.sql
   ```

3. **Setup Row Level Security (RLS)**
   - Enable RLS on all tables
   - Configure policies for user data access
   - The migration files include RLS setup

4. **Configure Authentication**
   - Go to Authentication → Settings
   - Configure your site URL
   - Enable email authentication
   - Optional: Setup OAuth providers (Google, GitHub, etc.)

---

## 🔐 Authentication Setup

### Supabase Auth Integration

1. **Configure Auth Settings**
   ```bash
   # In your .env file
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your-service-role-key  # Use service role key for server
   ```

2. **Client Integration Example**
   ```javascript
   // Example client-side auth setup
   import { createClient } from '@supabase/supabase-js'
   
   const supabase = createClient(
     'https://your-project.supabase.co',
     'your-anon-key'
   )
   
   // Sign up
   const { data, error } = await supabase.auth.signUp({
     email: 'user@example.com',
     password: 'password'
   })
   
   // Sign in
   const { data, error } = await supabase.auth.signInWithPassword({
     email: 'user@example.com',
     password: 'password'
   })
   ```

---

## 🤖 MCP Servers Setup

### Available MCP Servers

1. **Reddit MCP Server** (All tiers) - Port: 8001
2. **YouTube MCP Server** (All tiers) - Port: 8002
3. **GitHub MCP Server** (Pro/Enterprise only) - Port: 8003

### GitHub MCP Server Setup

1. **Create GitHub Personal Access Token**
   - Go to GitHub → Settings → Developer settings → Personal access tokens
   - Generate new token (classic)
   - Select scopes: `repo`, `read:org`, `read:user`
   - Copy token to `.env` file as `GITHUB_TOKEN`

2. **Start MCP Servers**
   ```bash
   # Start all MCP servers
   python scripts/start_reddit_server.py &
   python scripts/start_youtube_server.py &
   python scripts/start_github_server.py &
   ```

3. **Verify MCP Server Status**
   ```bash
   # Check if servers are running
   curl http://localhost:8001/health  # Reddit
   curl http://localhost:8002/health  # YouTube
   curl http://localhost:8003/health  # GitHub
   ```

---

## 🚀 Deployment Options

### Option 1: FastAPI Production Server

1. **Start the FastAPI Application**
   ```bash
   # Development
   python start_api.py
   
   # Production with Gunicorn
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api.main:app --bind 0.0.0.0:8000
   ```

2. **Access the API**
   ```
   - 🚀 API: http://localhost:8000
   - 📚 API Docs: http://localhost:8000/docs
   - 🔍 Health Check: http://localhost:8000/health
   ```

### Option 2: Docker Deployment

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.11-slim
   
   WORKDIR /app
   COPY . .
   
   RUN pip install -e .
   
   EXPOSE 8000
   CMD ["python", "start_api.py"]
   ```

2. **Build and Run**
   ```bash
   docker build -t open-deep-research .
   docker run -p 8000:8000 --env-file .env open-deep-research
   ```

---

## 📊 Subscription Tiers

### Free Tier (Default)
- ✅ 2 concurrent research units
- ✅ 1,000 API calls per day
- ✅ 20 research requests per day
- ✅ 50,000 tokens per day
- ✅ Reddit & YouTube MCP servers
- ✅ Basic support
- ❌ No GitHub MCP access
- ❌ No advanced RAG features

### Pro Tier ($29.99/month)
- ✅ 5 concurrent research units
- ✅ 10,000 API calls per day
- ✅ 200 research requests per day
- ✅ 500,000 tokens per day
- ✅ All MCP servers (Reddit, YouTube, GitHub)
- ✅ Advanced RAG features
- ✅ Priority support
- ✅ 500 RAG queries per day

### Enterprise Tier ($99.99/month)
- ✅ 20 concurrent research units
- ✅ Unlimited API calls
- ✅ Unlimited research requests
- ✅ Unlimited tokens
- ✅ All MCP servers
- ✅ Advanced RAG features
- ✅ Custom integrations
- ✅ Dedicated support
- ✅ Unlimited RAG queries

---

## 🔍 Troubleshooting

### Common Issues

#### 1. Stripe Webhook Issues
```bash
# Problem: Webhook events not received
# Solution: Check webhook endpoint URL and events selection

# Test webhook locally
stripe listen --forward-to localhost:8000/webhooks/stripe

# Verify webhook secret matches .env file
```

#### 2. Supabase Connection Issues
```bash
# Problem: Database connection failed
# Solution: Check URL and key in .env file

# Test connection
curl -H "apikey: YOUR_SUPABASE_KEY" \
     -H "Authorization: Bearer YOUR_SUPABASE_KEY" \
     "https://your-project.supabase.co/rest/v1/users"
```

#### 3. MCP Server Issues
```bash
# Problem: MCP servers not accessible
# Solution: Check if servers are running and ports are available

# Check server status
netstat -tulpn | grep :8001
netstat -tulpn | grep :8002
netstat -tulpn | grep :8003

# Restart servers
python scripts/start_reddit_server.py &
python scripts/start_youtube_server.py &
python scripts/start_github_server.py &
```

#### 4. GitHub API Issues
```bash
# Problem: GitHub MCP server authentication failed
# Solution: Check GitHub token permissions

# Test GitHub token
curl -H "Authorization: token YOUR_GITHUB_TOKEN" \
     https://api.github.com/user
```

#### 5. RAG System Issues
```bash
# Problem: Vector store not working
# Solution: Check pgvector extension and embeddings

# Verify pgvector extension
SELECT * FROM pg_extension WHERE extname = 'vector';

# Check OpenAI API key for embeddings
curl -H "Authorization: Bearer YOUR_OPENAI_KEY" \
     https://api.openai.com/v1/models
```

### API Endpoints for Testing

```bash
# Health check
GET /health

# Get subscription info
GET /subscription/info
Authorization: Bearer <jwt_token>

# Get usage statistics
GET /usage/stats?time_window=day
Authorization: Bearer <jwt_token>

# Test RAG search
POST /rag/search
Authorization: Bearer <jwt_token>
Content-Type: application/json
{
  "query": "test search",
  "k": 5
}

# Check MCP server access
GET /usage/mcp-access/github
Authorization: Bearer <jwt_token>
```

### Environment Variables Reference

```bash
# Required for basic functionality
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...

# Required for production features
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
STRIPE_PUBLISHABLE_KEY=pk_test_...
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_PRO_PRICE_ID=price_...
STRIPE_ENTERPRISE_PRICE_ID=price_...

# Optional
GITHUB_TOKEN=ghp_...
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=open-deep-research
```

### Configurations

Open Deep Research offers extensive configuration options to customize the research process and model behavior. All configurations can be set via the web UI, environment variables, or by modifying the configuration directly.

#### General Settings

- **Max Structured Output Retries** (default: 3): Maximum number of retries for structured output calls from models when parsing fails
- **Allow Clarification** (default: true): Whether to allow the researcher to ask clarifying questions before starting research
- **Max Concurrent Research Units** (default: 5): Maximum number of research units to run concurrently using sub-agents. Higher values enable faster research but may hit rate limits

#### Research Configuration

- **Search API** (default: Tavily): Choose from Tavily (works with all models), OpenAI Native Web Search, Anthropic Native Web Search, or None
- **Max Researcher Iterations** (default: 3): Number of times the Research Supervisor will reflect on research and ask follow-up questions
- **Max React Tool Calls** (default: 5): Maximum number of tool calling iterations in a single researcher step

#### Models

Open Deep Research uses multiple specialized models for different research tasks:

- **Summarization Model** (default: `openai:gpt-4.1-nano`): Summarizes research results from search APIs
- **Research Model** (default: `openai:gpt-4.1`): Conducts research and analysis 
- **Compression Model** (default: `openai:gpt-4.1-mini`): Compresses research findings from sub-agents
- **Final Report Model** (default: `openai:gpt-4.1`): Writes the final comprehensive report

All models are configured using [init_chat_model() API](https://python.langchain.com/docs/how_to/chat_models_universal_init/) which supports providers like OpenAI, Anthropic, Google Vertex AI, and others.

**Important Model Requirements:**

1. **Structured Outputs**: All models must support structured outputs. Check support [here](https://python.langchain.com/docs/integrations/chat/).

2. **Search API Compatibility**: Research and Compression models must support your selected search API:
   - Anthropic search requires Anthropic models with web search capability
   - OpenAI search requires OpenAI models with web search capability  
   - Tavily works with all models

3. **Tool Calling**: All models must support tool calling functionality

4. **Special Configurations**:
   - For OpenRouter: Follow [this guide](https://github.com/langchain-ai/open_deep_research/issues/75#issuecomment-2811472408)
   - For local models via Ollama: See [setup instructions](https://github.com/langchain-ai/open_deep_research/issues/65#issuecomment-2743586318)

#### Example MCP (Model Context Protocol) Servers

Open Deep Research supports MCP servers to extend research capabilities. 

#### Local MCP Servers

**Filesystem MCP Server** provides secure file system operations with robust access control:
- Read, write, and manage files and directories
- Perform operations like reading file contents, creating directories, moving files, and searching
- Restrict operations to predefined directories for security
- Support for both command-line configuration and dynamic MCP roots

Example usage:
```bash
mcp-server-filesystem /path/to/allowed/dir1 /path/to/allowed/dir2
```

#### Remote MCP Servers  

**Remote MCP servers** enable distributed agent coordination and support streamable HTTP requests. Unlike local servers, they can be multi-tenant and require more complex authentication.

**Arcade MCP Server Example**:
```json
{
  "url": "https://api.arcade.dev/v1/mcps/ms_0ujssxh0cECutqzMgbtXSGnjorm",
  "tools": ["Search_SearchHotels", "Search_SearchOneWayFlights", "Search_SearchRoundtripFlights"]
}
```

Remote servers can be configured as authenticated or unauthenticated and support JWT-based authentication through OAuth endpoints.

### Evaluation

A comprehensive batch evaluation system designed for detailed analysis and comparative studies.

#### **Features:**
- **Multi-dimensional Scoring**: Specialized evaluators with 0-1 scale ratings
- **Dataset-driven Evaluation**: Batch processing across multiple test cases

#### **Usage:**
```bash
# Run comprehensive evaluation on LangSmith datasets
python tests/run_evaluate.py
```
#### **Key Files:**
- `tests/run_evaluate.py`: Main evaluation script
- `tests/evaluators.py`: Specialized evaluator functions
- `tests/prompts.py`: Evaluation prompts for each dimension

### Deployments and Usages

#### LangGraph Studio

Follow the [quickstart](#-quickstart) to start LangGraph server locally and test the agent out on LangGraph Studio.

#### Hosted deployment
 
You can easily deploy to [LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/#deployment-options). 

#### Open Agent Platform

Open Agent Platform (OAP) is a UI from which non-technical users can build and configure their own agents. OAP is great for allowing users to configure the Deep Researcher with different MCP tools and search APIs that are best suited to their needs and the problems that they want to solve.

We've deployed Open Deep Research to our public demo instance of OAP. All you need to do is add your API Keys, and you can test out the Deep Researcher for yourself! Try it out [here](https://oap.langchain.com)

You can also deploy your own instance of OAP, and make your own custom agents (like Deep Researcher) available on it to your users.
1. [Deploy Open Agent Platform](https://docs.oap.langchain.com/quickstart)
2. [Add Deep Researcher to OAP](https://docs.oap.langchain.com/setup/agents)

### Updates 🔥

### Legacy Implementations 🏛️

The `src/legacy/` folder contains two earlier implementations that provide alternative approaches to automated research:

#### 1. Workflow Implementation (`legacy/graph.py`)
- **Plan-and-Execute**: Structured workflow with human-in-the-loop planning
- **Sequential Processing**: Creates sections one by one with reflection
- **Interactive Control**: Allows feedback and approval of report plans
- **Quality Focused**: Emphasizes accuracy through iterative refinement

#### 2. Multi-Agent Implementation (`legacy/multi_agent.py`)  
- **Supervisor-Researcher Architecture**: Coordinated multi-agent system
- **Parallel Processing**: Multiple researchers work simultaneously
- **Speed Optimized**: Faster report generation through concurrency
- **MCP Support**: Extensive Model Context Protocol integration

See `src/legacy/legacy.md` for detailed documentation, configuration options, and usage examples for both legacy implementations.


