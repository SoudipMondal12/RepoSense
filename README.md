# 🧠 RepoSense

> AI-powered GitHub repository intelligence using RAG, LangGraph, code relationships, Hugging Face embeddings, Chroma, and Groq.

RepoSense lets users connect a public GitHub repository and ask questions about its codebase using natural language.

It combines semantic retrieval with a code relationship graph so that answers can use both relevant code and relationships between functions, classes, imports, and callers.

---

## ✨ Features

- 🔗 Analyze any public GitHub repository
- 🔎 Semantic code search
- 🧠 Retrieval-Augmented Generation (RAG)
- 🕸️ Code relationship graph
- 🧭 LangGraph-based routing
- 💬 Conversational repository chat
- 🐛 Bug / debugging analysis
- 📚 Source file and line references
- ⚡ Groq-powered response generation
- 🤗 Hugging Face hosted embeddings
- 🗃️ Chroma vector store
- 🖥️ Streamlit web interface
- 🔐 Users provide their own API keys
- 📊 Repository telemetry and debugging logs

---

## 🏗️ Architecture

```text
                    ┌──────────────────────┐
                    │   Public GitHub Repo │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Repository Ingestion │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Code Parsing      │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    │                      │
                    ▼                      ▼
          ┌──────────────────┐   ┌────────────────────┐
          │ Hugging Face    │   │ Code Relationship  │
          │ Embeddings      │   │ Graph              │
          └────────┬─────────┘   └─────────┬──────────┘
                   │                       │
                   ▼                       │
          ┌──────────────────┐             │
          │     Chroma       │◄────────────┘
          │  Vector Store    │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │    LangGraph     │
          │ Router + RAG     │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │     Groq LLM     │
          └────────┬─────────┘
                   │
                   ▼
             ┌───────────┐
             │  Answer   │
             └───────────┘

🧩 Tech Stack

| Component          | Technology                               |
| ------------------ | ---------------------------------------- |
| UI                 | Streamlit                                |
| LLM                | Groq                                     |
| Embeddings         | Hugging Face Inference API               |
| Embedding model    | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database    | Chroma                                   |
| Orchestration      | LangGraph                                |
| RAG                | LangChain                                |
| Repository source  | GitHub API                               |
| Code analysis      | Python AST / repository parser           |
| Code relationships | NetworkX-based graph                     |
| Language           | Python                                   |

🚀 Live Application



📦 Project Structure

RepoScense/
│
├── app/
│   │
│   ├── streamlit_app.py
│   │
│   ├── embeddings/
│   │   └── huggingface.py
│   │
│   ├── github/
│   │   └── client.py
│   │
│   ├── ingestion/
│   │   └── repository.py
│   │
│   ├── retrieval/
│   │   ├── indexer.py
│   │   └── vector_store.py
│   │
│   ├── analysis/
│   │   ├── code_parser.py
│   │   └── relationship_graph.py
│   │
│   ├── graph/
│   │   ├── state.py
│   │   ├── router.py
│   │   ├── nodes.py
│   │   └── workflow.py
│   │
│   ├── rag/
│   │   ├── chain.py
│   │   └── context.py
│   │
│   └── llm/
│       └── groq.py
│
├── tests/
│
├── requirements.txt
│
├── .gitignore
│
└── README.md

🔑 API Keys

RepoSense requires:

Hugging Face API key
Groq API key

A GitHub token is optional when analyzing public repositories.

Users enter these credentials directly into the RepoSense interface.

🤗 Hugging Face API Key 

RepoSense uses the Hugging Face Inference API for text embeddings.

How to create a Hugging Face token
1. Create or sign in to your Hugging Face account.

2. Open:

https://huggingface.co/settings/tokens

3. Create a new token.
4. Give it the required Inference Providers permission.
5. Copy the token.

Hugging Face documents the Feature Extraction API for converting text into embeddings and requires an access token with the appropriate Inference Providers permission.
Reference:

https://huggingface.co/docs/inference-providers/tasks/feature-extraction


⚡ Groq API Key

RepoSense uses Groq for LLM generation.

How to create a Groq API key
1. Create or sign in to your Groq account.
2. Open the Groq Console.
3. Go to the API Keys section.
4. Create a new API key.
5. Copy the key.

Groq's official quickstart provides the API-key creation flow:

https://console.groq.com/docs/quickstart

🐙 GitHub Repository

Enter a public repository URL such as:

https://github.com/username/repository

🛠️ Local Installation

1. Clone the repository

git clone https://github.com/YOUR_USERNAME/RepoScense.git
cd RepoScense

2. Create a virtual environment

Windows

python -m venv .venv

Activate it:

.venv\Scripts\Activate.ps1

Linux / macOS

python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Start RepoSense

Run from the repository root:

streamlit run app/streamlit_app.py

💬 Example Questions

Once a repository has been indexed, try:

What is the main purpose of this repository?
Where is authentication implemented?
Explain the main entry point.
How does the authentication flow work?
Explain the AgentRegistry class.
Where is the database connection created?
Find a potential bug in the login flow.
What files depend on this function?
How would you improve this implementation?

You can also ask follow-up questions such as:

Explain that function in more detail.
What happens if that function fails?
What happens if that function fails?

🔄 RAG Pipeline

RepoSense follows this general retrieval flow:

User Question
      ↓
LangGraph Router
      ↓
Retrieval Query
      ↓
Hugging Face Embedding
      ↓
Chroma Similarity / Hybrid Search
      ↓
Relevant Code Documents
      ↓
Code Relationship Expansion
      ↓
Context Limiting
      ↓
Groq LLM
      ↓
Grounded Answer

The system also uses bounded evidence/context before sending repository information to the LLM to reduce oversized requests and improve reliability.

🕸️ Code Relationship Graph

RepoSense builds a graph of relationships within the repository.

Examples include:

imports
calls
contains

This allows retrieval to go beyond simple semantic similarity.

For example:

authenticate_user()
        │
        ├── calls → get_user()
        │
        ├── imports → database.users
        │
        └── contains → validation logic

This related-code evidence can be used during explanation and debugging.

🐛 Debugging Mode

RepoSense can analyze debugging questions by combining:

>semantic retrieval
>related code
>callers/dependencies
>repository evidence
>constrained LLM reasoning

The application distinguishes between levels such as:

CONFIRMED
LIKELY
POSSIBLE
INSUFFICIENT_EVIDENCE

A bug should not be considered confirmed without supporting repository evidence.


## 🖥️ Application Screenshots

![RepoSense Screenshot 1](./screenshot-1.png)

![RepoSense Screenshot 2](./screenshot-2.png)
