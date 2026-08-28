from __future__ import annotations

import sys
import time
import traceback
import uuid
from pathlib import Path

import streamlit as st

from langchain_core.documents import Document


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# PROJECT IMPORTS
# ============================================================

from app.github.client import GitHubClient

from app.ingestion.repository import (
    RepositoryIngestor,
)

from app.embeddings.huggingface import (
    HuggingFaceEmbeddingService,
)

from app.retrieval.indexer import (
    RepositoryIndexer,
)

from app.analysis.relationship_graph import (
    CodeRelationshipGraph,
)

from app.graph.workflow import (
    build_repo_sense_graph,
)


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

HF_MODEL = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

VECTORSTORE_DIRECTORY = (
    "data/vectorstore_huggingface"
)

MAX_REPOSITORY_FILES = 100


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title="RepoSense",

    page_icon="🧠",

    layout="wide",

    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

@import url(
'https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap'
);


/* ============================================================
   GLOBAL
   ============================================================ */

html,
body,
[class*="css"] {

    font-family:
        "JetBrains Mono",
        monospace !important;
}

.stApp {

    background:
        radial-gradient(
            circle at 12% 8%,
            rgba(
                0,
                255,
                170,
                0.05
            ),
            transparent 25%
        ),

        radial-gradient(
            circle at 90% 90%,
            rgba(
                0,
                150,
                255,
                0.025
            ),
            transparent 25%
        ),

        #030604;

    color:
        #d9ffe9;
}


.block-container {

    max-width:
        1450px;

    padding-top:
        1rem;

    padding-bottom:
        3rem;
}


/* ============================================================
   STREAMLIT CLEANUP
   ============================================================ */

#MainMenu {

    visibility:
        hidden;
}

footer {

    visibility:
        hidden;
}

header {

    background:
        transparent !important;
}


/* ============================================================
   SCROLLBAR
   ============================================================ */

::-webkit-scrollbar {

    width:
        7px;

    height:
        7px;
}

::-webkit-scrollbar-track {

    background:
        #020503;
}

::-webkit-scrollbar-thumb {

    background:
        #16462d;

    border-radius:
        5px;
}

::-webkit-scrollbar-thumb:hover {

    background:
        #286b46;
}


/* ============================================================
   HEADER
   ============================================================ */

.header-box {

    background:
        linear-gradient(
            135deg,
            #07110b,
            #050a07
        );

    border:
        1px solid
        rgba(
            0,
            255,
            170,
            0.15
        );

    border-radius:
        10px;

    padding:
        16px 20px;

    margin-bottom:
        20px;
}


.brand {

    color:
        #9cffc6;

    font-size:
        23px;

    font-weight:
        700;

    letter-spacing:
        3px;
}


.brand-subtitle {

    color:
        #526d5d;

    font-size:
        8px;

    margin-top:
        6px;

    letter-spacing:
        1.6px;
}


.online {

    color:
        #65ffb5;

    font-size:
        8px;

    font-weight:
        700;

    letter-spacing:
        1px;

    text-align:
        right;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {

    background:
        #050a07;

    border-right:
        1px solid
        rgba(
            0,
            255,
            170,
            0.13
        );
}

section[data-testid="stSidebar"]
> div {

    padding-top:
        1.25rem;
}


.sidebar-brand {

    color:
        #76ffb7;

    font-size:
        17px;

    font-weight:
        700;

    letter-spacing:
        2px;
}


.sidebar-subtitle {

    color:
        #506b5b;

    font-size:
        8px;

    margin-top:
        4px;

    letter-spacing:
        1px;
}


.sidebar-section {

    color:
        #4e6959;

    font-size:
        8px;

    font-weight:
        700;

    letter-spacing:
        2px;

    margin-top:
        22px;

    margin-bottom:
        9px;
}


.sidebar-status {

    color:
        #748b7e;

    font-size:
        8px;

    line-height:
        1.95;
}


.sidebar-help {

    color:
        #50675a;

    font-size:
        8px;

    line-height:
        1.8;
}


/* ============================================================
   HERO
   ============================================================ */

.hero {

    text-align:
        center;

    padding:
        45px 10px 35px;
}


.hero-title {

    color:
        #a5ffcb;

    font-size:
        38px;

    font-weight:
        700;

    letter-spacing:
        5px;

    text-shadow:
        0 0 28px
        rgba(
            0,
            255,
            170,
            0.15
        );
}


.hero-subtitle {

    color:
        #5d7667;

    font-size:
        9px;

    letter-spacing:
        2px;

    margin-top:
        12px;
}


.hero-command {

    color:
        #344b3e;

    font-size:
        8px;

    letter-spacing:
        1.5px;

    margin-top:
        17px;
}


/* ============================================================
   SECTION
   ============================================================ */

.section-title {

    color:
        #65ffb5;

    font-size:
        9px;

    font-weight:
        700;

    letter-spacing:
        2px;

    border-bottom:
        1px solid
        rgba(
            0,
            255,
            170,
            0.11
        );

    padding-bottom:
        8px;

    margin-top:
        22px;

    margin-bottom:
        14px;
}


/* ============================================================
   CAPABILITY CARD
   ============================================================ */

.capability-card {

    background:
        linear-gradient(
            145deg,
            #08130d,
            #050a07
        );

    border:
        1px solid
        rgba(
            0,
            255,
            170,
            0.14
        );

    border-radius:
        10px;

    min-height:
        170px;

    padding:
        20px;
}


.card-number {

    color:
        #315e46;

    font-size:
        9px;

    font-weight:
        700;

    letter-spacing:
        2px;

    margin-bottom:
        17px;
}


.card-title {

    color:
        #a0ffca;

    font-size:
        10px;

    font-weight:
        700;

    letter-spacing:
        1px;

    margin-bottom:
        12px;
}


.card-description {

    color:
        #60796a;

    font-size:
        8px;

    line-height:
        1.75;
}


/* ============================================================
   STATUS BADGE
   ============================================================ */

.status-badge {

    display:
        inline-block;

    padding:
        5px 8px;

    border:
        1px solid
        rgba(
            0,
            255,
            170,
            0.16
        );

    border-radius:
        5px;

    background:
        rgba(
            0,
            255,
            170,
            0.035
        );

    color:
        #63e99f;

    font-size:
        7px;

    letter-spacing:
        1px;
}


/* ============================================================
   INPUTS
   ============================================================ */

.stTextInput input {

    background:
        #060d09 !important;

    color:
        #baffd4 !important;

    border:
        1px solid
        rgba(
            0,
            255,
            170,
            0.21
        ) !important;

    border-radius:
        7px !important;

    font-family:
        "JetBrains Mono",
        monospace !important;

    font-size:
        9px !important;
}


.stTextInput input:focus {

    border-color:
        rgba(
            0,
            255,
            170,
            0.60
        ) !important;

    box-shadow:
        0 0 14px
        rgba(
            0,
            255,
            170,
            0.07
        ) !important;
}


/* ============================================================
   BUTTONS
   ============================================================ */

.stButton > button {

    background:
        #09160f;

    color:
        #72ffb8;

    border:
        1px solid
        rgba(
            0,
            255,
            170,
            0.28
        );

    border-radius:
        7px;

    font-family:
        "JetBrains Mono",
        monospace;

    font-size:
        8px;

    font-weight:
        700;

    letter-spacing:
        1px;

    min-height:
        37px;
}


.stButton > button:hover {

    background:
        #0c2016;

    color:
        #a5ffd0;

    border-color:
        rgba(
            0,
            255,
            170,
            0.60
        );

    box-shadow:
        0 0 18px
        rgba(
            0,
            255,
            170,
            0.07
        );
}


/* ============================================================
   CHAT
   ============================================================ */

div[data-testid="stChatMessage"] {

    background:
        #070e0a;

    border:
        1px solid
        rgba(
            0,
            255,
            170,
            0.09
        );

    border-radius:
        9px;

    margin-bottom:
        9px;
}


.stChatInput textarea {

    background:
        #060d09 !important;

    color:
        #baffd4 !important;

    border:
        1px solid
        rgba(
            0,
            255,
            170,
            0.22
        ) !important;

    font-family:
        "JetBrains Mono",
        monospace !important;

    font-size:
        9px !important;
}


/* ============================================================
   METRICS
   ============================================================ */

div[data-testid="stMetric"] {

    background:
        #070e0a;

    border:
        1px solid
        rgba(
            0,
            255,
            170,
            0.12
        );

    border-radius:
        8px;

    padding:
        10px;
}


div[data-testid="stMetricLabel"] {

    color:
        #536d5f !important;

    font-size:
        7px !important;
}


div[data-testid="stMetricValue"] {

    color:
        #a5ffcb !important;
}


/* ============================================================
   EXPANDERS
   ============================================================ */

.streamlit-expanderHeader {

    background:
        #070d0a !important;

    color:
        #73b992 !important;

    font-family:
        "JetBrains Mono",
        monospace !important;

    font-size:
        8px !important;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {

    text-align:
        center;

    color:
        #2d4136;

    font-size:
        7px;

    letter-spacing:
        1.5px;

    line-height:
        2;

    margin-top:
        40px;

    padding-top:
        20px;

    border-top:
        1px solid
        rgba(
            0,
            255,
            170,
            0.07
        );
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {

    "initialized": False,

    "repo_url": "",

    "owner": "",

    "repo": "",

    "branch": "",

    "description": "",

    "documents": [],

    "indexer": None,

    "code_graph": None,

    "graph": None,

    "graph_summary": {},

    "indexed_count": 0,

    "messages": [],

    "logs": [],

    "thread_id": "",

    "hf_token": "",

    "groq_api_key": "",
}


for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# HELPERS
# ============================================================

def add_log(
    message: str,
) -> None:

    timestamp = time.strftime(
        "%H:%M:%S"
    )

    st.session_state.logs.append(

        f"[{timestamp}] {message}"
    )

    st.session_state.logs = (
        st.session_state.logs[-100:]
    )


def create_thread_id(
    owner: str,
    repo: str,
    branch: str,
) -> str:

    # A new thread per repository analysis.
    # This prevents conversations between
    # different repositories from mixing.

    safe_owner = (
        owner
        .replace(
            "/",
            "-",
        )
        .replace(
            " ",
            "-",
        )
    )

    safe_repo = (
        repo
        .replace(
            "/",
            "-",
        )
        .replace(
            " ",
            "-",
        )
    )

    safe_branch = (
        branch
        .replace(
            "/",
            "-",
        )
        .replace(
            " ",
            "-",
        )
    )

    random_part = (
        uuid.uuid4()
        .hex[:8]
    )

    return (
        "reposense-"
        f"{safe_owner}-"
        f"{safe_repo}-"
        f"{safe_branch}-"
        f"{random_part}"
    )


def convert_documents(
    code_documents,
) -> list[Document]:

    documents = []

    for code_document in code_documents:

        content = getattr(
            code_document,
            "content",
            "",
        )

        if not content:

            continue


        metadata = {

            "repository":
                getattr(
                    code_document,
                    "repository",
                    "",
                ),

            "file_path":
                getattr(
                    code_document,
                    "file_path",
                    "",
                ),

            "symbol_name":
                getattr(
                    code_document,
                    "symbol_name",
                    "",
                ),

            "symbol_type":
                getattr(
                    code_document,
                    "symbol_type",
                    "",
                ),

            "parent_symbol":
                getattr(
                    code_document,
                    "parent_symbol",
                    "",
                ),

            "language":
                getattr(
                    code_document,
                    "language",
                    "",
                ),

            "start_line":
                getattr(
                    code_document,
                    "start_line",
                    None,
                ),

            "end_line":
                getattr(
                    code_document,
                    "end_line",
                    None,
                ),

            "document_type":
                getattr(
                    code_document,
                    "document_type",
                    "",
                ),
        }


        documents.append(

            Document(

                page_content=str(
                    content
                ),

                metadata=metadata,
            )
        )


    return documents


def reset_repository():

    st.session_state.initialized = False

    st.session_state.repo_url = ""

    st.session_state.owner = ""

    st.session_state.repo = ""

    st.session_state.branch = ""

    st.session_state.description = ""

    st.session_state.documents = []

    st.session_state.indexer = None

    st.session_state.code_graph = None

    st.session_state.graph = None

    st.session_state.graph_summary = {}

    st.session_state.indexed_count = 0

    st.session_state.messages = []

    st.session_state.logs = []

    st.session_state.thread_id = ""


# ============================================================
# HEADER
# ============================================================

header_left, header_right = (
    st.columns([5, 1])
)


with header_left:

    st.markdown(
        '<div class="header-box">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="brand">'
        '🧠 REPOSENSE'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="brand-subtitle">'
        'AI CODE INTELLIGENCE // '
        'RAG + LANGGRAPH + CODE GRAPH'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


with header_right:

    st.markdown(
        '<div class="header-box">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="online">'
        '● SYSTEM ONLINE'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">'
        '◉ REPOSENSE'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'REPOSITORY INTELLIGENCE ENGINE'
        '</div>',
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # Repository URL
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">'
        'TARGET REPOSITORY'
        '</div>',
        unsafe_allow_html=True,
    )


    user_repo_url = st.text_input(

        "GitHub Repository URL",

        value=(
            st.session_state.repo_url
        ),

        placeholder=(
            "https://github.com/user/repository"
        ),

        label_visibility="collapsed",
    )


    # --------------------------------------------------------
    # Hugging Face API
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">'
        'HUGGING FACE API'
        '</div>',
        unsafe_allow_html=True,
    )


    user_hf_token = st.text_input(

        "Hugging Face API Key",

        value=(
            st.session_state.hf_token
        ),

        type="password",

        placeholder="hf_...",

        label_visibility="collapsed",
    )


    # --------------------------------------------------------
    # Groq API
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">'
        'GROQ API'
        '</div>',
        unsafe_allow_html=True,
    )


    user_groq_key = st.text_input(

        "Groq API Key",

        value=(
            st.session_state.groq_api_key
        ),

        type="password",

        placeholder="gsk_...",

        label_visibility="collapsed",
    )


    # --------------------------------------------------------
    # Optional GitHub token
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">'
        'GITHUB TOKEN'
        '</div>',
        unsafe_allow_html=True,
    )


    user_github_token = st.text_input(

        "GitHub Token",

        value="",

        type="password",

        placeholder="Optional",

        label_visibility="collapsed",
    )


    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    analyze_button = st.button(

        "⚡ ANALYZE REPOSITORY",

        use_container_width=True,
    )


    # --------------------------------------------------------
    # Change repository
    # --------------------------------------------------------

    if st.session_state.initialized:

        if st.button(

            "↻ CHANGE REPOSITORY",

            use_container_width=True,
        ):

            reset_repository()

            st.rerun()


    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">'
        'PIPELINE'
        '</div>',
        unsafe_allow_html=True,
    )


    st.markdown(
        '<div class="sidebar-status">'
        '● GitHub repository ingestion<br>'
        '● Source-code parsing<br>'
        '● Hugging Face embeddings<br>'
        '● Chroma vector retrieval<br>'
        '● Code relationship graph<br>'
        '● LangGraph routing<br>'
        '● Groq generation'
        '</div>',
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # Services
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">'
        'SERVICE STATUS'
        '</div>',
        unsafe_allow_html=True,
    )


    if user_hf_token:

        st.markdown(
            '<span class="status-badge">'
            '● HF KEY PROVIDED'
            '</span>',
            unsafe_allow_html=True,
        )

    else:

        st.caption(
            "○ HF key not provided"
        )


    if user_groq_key:

        st.markdown(
            '<span class="status-badge">'
            '● GROQ KEY PROVIDED'
            '</span>',
            unsafe_allow_html=True,
        )

    else:

        st.caption(
            "○ Groq key not provided"
        )


    if user_github_token:

        st.markdown(
            '<span class="status-badge">'
            '● GITHUB TOKEN PROVIDED'
            '</span>',
            unsafe_allow_html=True,
        )

    else:

        st.caption(
            "● GitHub public mode"
        )


    # --------------------------------------------------------
    # Help
    # --------------------------------------------------------

    with st.expander(
        "HOW TO GET API KEYS"
    ):

        st.markdown(
            """
**Hugging Face**

1. Create/sign in to your Hugging Face account.
2. Open **Settings → Access Tokens**.
3. Create a token with the required
   **Inference Providers** permission.
4. Copy the token beginning with `hf_`.

**Groq**

1. Create/sign in to your Groq account.
2. Open the API Keys section.
3. Create a new API key.
4. Copy the key beginning with `gsk_`.

**GitHub**

A GitHub token is optional for public
repositories. It can be useful when you
need higher GitHub API limits.

**Security**

Never paste API keys into your GitHub
repository or source code.

The keys entered here are used by the
current Streamlit session.
"""
        )


# ============================================================
# ANALYZE REPOSITORY
# ============================================================

if analyze_button:

    repo_url = (
        user_repo_url.strip()
    )

    hf_token = (
        user_hf_token.strip()
    )

    groq_api_key = (
        user_groq_key.strip()
    )

    github_token = (
        user_github_token.strip()
        or None
    )


    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not repo_url:

        st.error(
            "Enter a GitHub repository URL."
        )

        st.stop()


    if not hf_token:

        st.error(
            "Enter your Hugging Face API key."
        )

        st.stop()


    if not groq_api_key:

        st.error(
            "Enter your Groq API key."
        )

        st.stop()


    # --------------------------------------------------------
    # Save keys to current session
    # --------------------------------------------------------

    st.session_state.repo_url = repo_url

    st.session_state.hf_token = hf_token

    st.session_state.groq_api_key = (
        groq_api_key
    )


    # --------------------------------------------------------
    # Reset previous repository
    # --------------------------------------------------------

    st.session_state.initialized = False

    st.session_state.messages = []

    st.session_state.logs = []

    st.session_state.documents = []

    st.session_state.indexer = None

    st.session_state.code_graph = None

    st.session_state.graph = None

    st.session_state.graph_summary = {}

    st.session_state.indexed_count = 0

    st.session_state.thread_id = ""


    # --------------------------------------------------------
    # Initialization UI
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'REPOSITORY INITIALIZATION'
        '</div>',
        unsafe_allow_html=True,
    )


    progress = st.progress(
        0,
        text="Starting RepoSense...",
    )


    status = st.empty()


    try:

        # ====================================================
        # 01 — GITHUB CONNECTION
        # ====================================================

        status.write(
            "▸ Connecting to GitHub..."
        )


        add_log(
            "Connecting to GitHub"
        )


        github = GitHubClient(
            token=github_token
        )


        owner, repo = (
            github.parse_repo_url(
                repo_url
            )
        )


        st.session_state.owner = owner

        st.session_state.repo = repo


        add_log(
            f"Repository: "
            f"{owner}/{repo}"
        )


        progress.progress(
            10,
            text="Repository detected",
        )


        # ====================================================
        # 02 — REPOSITORY METADATA
        # ====================================================

        status.write(
            "▸ Reading repository metadata..."
        )


        repository_info = (
            github.get_repository(
                owner,
                repo,
            )
        )


        branch = (
            repository_info[
                "default_branch"
            ]
        )


        description = (
            repository_info.get(
                "description",
                "",
            )
            or ""
        )


        st.session_state.branch = branch

        st.session_state.description = (
            description
        )


        # ----------------------------------------------------
        # Create conversation thread
        # ----------------------------------------------------

        st.session_state.thread_id = (
            create_thread_id(
                owner,
                repo,
                branch,
            )
        )


        add_log(
            f"Default branch: {branch}"
        )


        add_log(
            "Thread created: "
            f"{st.session_state.thread_id}"
        )


        progress.progress(
            20,
            text="Repository metadata loaded",
        )


        # ====================================================
        # 03 — REPOSITORY INGESTION
        # ====================================================

        status.write(
            "▸ Ingesting repository source code..."
        )


        add_log(
            "Starting repository ingestion"
        )


        ingestor = RepositoryIngestor(
            github
        )


        code_documents = (
            ingestor.ingest(

                owner=owner,

                repo=repo,

                branch=branch,

                max_files=(
                    MAX_REPOSITORY_FILES
                ),
            )
        )


        if not code_documents:

            raise RuntimeError(
                "No processable source-code "
                "documents were found."
            )


        st.session_state.documents = (
            code_documents
        )


        add_log(
            "Ingestion complete: "
            f"{len(code_documents)} "
            "CodeDocuments"
        )


        progress.progress(
            40,
            text=(
                f"Parsed "
                f"{len(code_documents)} "
                "code documents"
            ),
        )


        # ====================================================
        # 04 — CODE RELATIONSHIP GRAPH
        # ====================================================

        status.write(
            "▸ Building code relationship graph..."
        )


        add_log(
            "Building code relationship graph"
        )


        code_graph = (
            CodeRelationshipGraph(
                code_documents
            )
        )


        graph_summary = (
            code_graph.summary()
        )


        st.session_state.code_graph = (
            code_graph
        )

        st.session_state.graph_summary = (
            graph_summary
        )


        add_log(
            "Graph nodes: "
            f"{graph_summary.get('nodes', 0)}"
        )


        add_log(
            "Graph relationships: "
            f"{graph_summary.get('relationships', 0)}"
        )


        progress.progress(
            55,
            text="Code graph ready",
        )


        # ====================================================
        # 05 — HUGGING FACE EMBEDDINGS
        # ====================================================

        status.write(
            "▸ Initializing Hugging Face embeddings..."
        )


        add_log(
            "Initializing Hugging Face embedding service"
        )


        add_log(
            f"Embedding model: {HF_MODEL}"
        )


        embedding_service = (
            HuggingFaceEmbeddingService(

                api_key=hf_token,

                model=HF_MODEL,

                batch_size=4,

                max_retries=3,

                retry_delay=3.0,

                timeout=120,
            )
        )


        add_log(
            "Hugging Face embedding service ready"
        )


        progress.progress(
            62,
            text="Embedding service ready",
        )


        # ====================================================
        # 06 — CONVERT DOCUMENTS
        # ====================================================

        status.write(
            "▸ Preparing documents for indexing..."
        )


        langchain_documents = (
            convert_documents(
                code_documents
            )
        )


        if not langchain_documents:

            raise RuntimeError(
                "No LangChain documents were "
                "created from the repository."
            )


        add_log(
            "Converted "
            f"{len(langchain_documents)} "
            "documents"
        )


        progress.progress(
            67,
            text="Documents prepared",
        )


        # ====================================================
        # 07 — VECTOR INDEX
        # ====================================================

        status.write(
            "▸ Building Chroma vector index..."
        )


        add_log(
            "Creating RepositoryIndexer"
        )


        add_log(
            "Vector store: "
            f"{VECTORSTORE_DIRECTORY}"
        )


        indexer = RepositoryIndexer(

            embedding_service=(
                embedding_service
            ),

            owner=owner,

            repo=repo,

            branch=branch,

            persist_directory=(
                VECTORSTORE_DIRECTORY
            ),
        )


        add_log(
            "RepositoryVectorStore initialized"
        )


        indexed_count = (
            indexer.index(

                documents=(
                    langchain_documents
                ),

                clear_existing=True,
            )
        )


        if indexed_count <= 0:

            raise RuntimeError(
                "Vector indexing returned zero "
                "documents."
            )


        st.session_state.indexer = (
            indexer
        )

        st.session_state.indexed_count = (
            indexed_count
        )


        add_log(
            "Vector index ready: "
            f"{indexed_count} documents"
        )


        progress.progress(
            80,
            text=(
                f"Indexed "
                f"{indexed_count} documents"
            ),
        )


        # ====================================================
        # 08 — LANGGRAPH
        # ====================================================

        status.write(
            "▸ Compiling LangGraph..."
        )


        add_log(
            "Building LangGraph workflow"
        )


        graph = build_repo_sense_graph(

            api_key=groq_api_key,

            indexer=indexer,

            code_graph=code_graph,
        )


        st.session_state.graph = (
            graph
        )


        add_log(
            "LangGraph compiled successfully"
        )


        progress.progress(
            95,
            text="LangGraph ready",
        )


        # ====================================================
        # 09 — COMPLETE
        # ====================================================

        st.session_state.initialized = (
            True
        )


        add_log(
            "Repository initialization complete"
        )


        progress.progress(
            100,
            text="Repository analysis complete",
        )


        time.sleep(
            0.3
        )


        progress.empty()

        status.empty()


        st.success(
            f"✓ {owner}/{repo} is ready."
        )


        st.rerun()


    except Exception as exc:

        progress.empty()

        status.empty()


        add_log(
            "INITIALIZATION ERROR: "
            f"{type(exc).__name__}: {exc}"
        )


        st.error(
            "Repository initialization failed: "
            f"{type(exc).__name__}: {exc}"
        )


        with st.expander(
            "▼ DEBUG TRACE"
        ):

            st.code(
                traceback.format_exc(),
                language="text",
            )


        st.stop()


# ============================================================
# LANDING PAGE
# ============================================================

if not st.session_state.initialized:

    st.markdown(
        '<div class="hero">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-title">'
        'REPOSITORY INTELLIGENCE'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-subtitle">'
        'CONNECT ANY PUBLIC GITHUB REPOSITORY'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-command">'
        'RAG // LANGGRAPH // CODE GRAPH // '
        'HUGGING FACE // GROQ'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


    st.markdown(
        '<div class="section-title">'
        'SYSTEM CAPABILITIES'
        '</div>',
        unsafe_allow_html=True,
    )


    col1, col2, col3, col4 = (
        st.columns(4)
    )


    cards = [

        (
            col1,
            "01",
            "SEMANTIC SEARCH",
            "Find relevant files, functions and code using natural-language questions.",
        ),

        (
            col2,
            "02",
            "CODE EXPLANATION",
            "Understand functions, classes, modules and repository architecture.",
        ),

        (
            col3,
            "03",
            "BUG ANALYSIS",
            "Investigate potential problems using retrieval and code relationships.",
        ),

        (
            col4,
            "04",
            "REPOSITORY CHAT",
            "Ask follow-up questions about the indexed repository.",
        ),
    ]


    for (
        column,
        number,
        title,
        description,
    ) in cards:

        with column:

            st.markdown(
                '<div class="capability-card">',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="card-number">'
                f'{number}'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="card-title">'
                f'{title}'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="card-description">'
                f'{description}'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True,
            )


    st.markdown(
        '<div class="footer">'
        'ENTER YOUR GITHUB URL + API KEYS '
        'IN THE LEFT PANEL'
        '<br><br>'
        'RAG · CHROMA · LANGGRAPH · GROQ'
        '</div>',
        unsafe_allow_html=True,
    )


    st.stop()


# ============================================================
# ACTIVE REPOSITORY
# ============================================================

st.markdown(
    '<div class="section-title">'
    'ACTIVE REPOSITORY'
    '</div>',
    unsafe_allow_html=True,
)


st.code(

    f"$ repository\n"
    f"{st.session_state.owner}/"
    f"{st.session_state.repo}\n\n"
    f"branch    : "
    f"{st.session_state.branch}\n"
    f"status    : INDEXED\n"
    f"embedding : "
    f"{HF_MODEL}\n"
    f"documents : "
    f"{len(st.session_state.documents)}\n"
    f"vectors   : "
    f"{st.session_state.indexed_count}",

    language="text",
)


if st.session_state.description:

    st.caption(
        st.session_state.description
    )


# ============================================================
# TELEMETRY
# ============================================================

st.markdown(
    '<div class="section-title">'
    'REPOSITORY TELEMETRY'
    '</div>',
    unsafe_allow_html=True,
)


summary = (
    st.session_state.graph_summary
)


metric1, metric2, metric3, metric4 = (
    st.columns(4)
)


with metric1:

    st.metric(
        "CODE DOCUMENTS",
        len(
            st.session_state.documents
        ),
    )


with metric2:

    st.metric(
        "VECTOR INDEX",
        st.session_state.indexed_count,
    )


with metric3:

    st.metric(
        "GRAPH NODES",
        summary.get(
            "nodes",
            0,
        ),
    )


with metric4:

    st.metric(
        "RELATIONSHIPS",
        summary.get(
            "relationships",
            0,
        ),
    )


# ============================================================
# RELATIONSHIP TYPES
# ============================================================

relationship_types = (
    summary.get(
        "relationship_types",
        {},
    )
)


if relationship_types:

    st.markdown(
        '<div class="section-title">'
        'CODE RELATIONSHIPS'
        '</div>',
        unsafe_allow_html=True,
    )


    relationship_columns = (
        st.columns(
            min(
                len(
                    relationship_types
                ),
                4,
            )
        )
    )


    for index, (
        relationship,
        count,
    ) in enumerate(
        relationship_types.items()
    ):

        with relationship_columns[
            index
            % len(
                relationship_columns
            )
        ]:

            st.metric(
                relationship.upper(),
                count,
            )


# ============================================================
# CONVERSATION STATUS
# ============================================================

st.markdown(
    '<div class="section-title">'
    'CONVERSATION STATUS'
    '</div>',
    unsafe_allow_html=True,
)


st.caption(
    "Current repository conversation thread:"
)


st.code(
    st.session_state.thread_id,
    language="text",
)


# ============================================================
# CHAT
# ============================================================

st.markdown(
    '<div class="section-title">'
    'REPOSITORY CHAT'
    '</div>',
    unsafe_allow_html=True,
)


st.info(
    "Ask about architecture, functions, "
    "classes, authentication, dependencies, "
    "bugs, data flow, implementation details, "
    "or any part of the repository."
)


# ============================================================
# CHAT HISTORY
# ============================================================

for message in (
    st.session_state.messages
):

    role = message.get(
        "role",
        "assistant",
    )


    content = message.get(
        "content",
        "",
    )


    with st.chat_message(
        role
    ):

        st.markdown(
            content
        )


        if role == "assistant":

            intent = message.get(
                "intent"
            )

            duration = message.get(
                "duration"
            )


            metadata = []


            if intent:

                metadata.append(
                    f"intent={intent}"
                )


            if duration is not None:

                metadata.append(
                    f"time={duration:.2f}s"
                )


            if metadata:

                st.caption(
                    " · ".join(
                        metadata
                    )
                )


            sources = message.get(
                "sources",
                [],
            )


            if sources:

                with st.expander(
                    f"VIEW SOURCES "
                    f"({len(sources)})"
                ):

                    for source in sources:

                        file_path = (
                            source.get(
                                "file_path",
                                "Unknown",
                            )
                        )


                        symbol_name = (
                            source.get(
                                "symbol_name",
                                "",
                            )
                        )


                        start_line = (
                            source.get(
                                "start_line",
                                "?",
                            )
                        )


                        end_line = (
                            source.get(
                                "end_line",
                                "?",
                            )
                        )


                        st.markdown(
                            f"**◈ {file_path}**"
                        )


                        if symbol_name:

                            st.caption(
                                f"Symbol: "
                                f"{symbol_name}"
                            )


                        st.caption(
                            f"Lines: "
                            f"{start_line}-"
                            f"{end_line}"
                        )


                        st.divider()


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "ask RepoSense about the code..."
)


if question:

    question = question.strip()


    if not question:

        st.stop()


    if st.session_state.graph is None:

        st.error(
            "LangGraph is not initialized."
        )

        st.stop()


    # --------------------------------------------------------
    # Save user message
    # --------------------------------------------------------

    st.session_state.messages.append(

        {
            "role":
                "user",

            "content":
                question,
        }
    )


    add_log(
        f"Question: {question}"
    )


    # ========================================================
    # LANGGRAPH EXECUTION
    # ========================================================

    with st.spinner(
        "RepoSense is analyzing the repository..."
    ):

        start_time = (
            time.perf_counter()
        )


        try:

            # ------------------------------------------------
            # CHECKPOINTER THREAD
            # ------------------------------------------------

            thread_id = (
                st.session_state.get(
                    "thread_id"
                )
            )


            if not thread_id:

                thread_id = (
                    create_thread_id(

                        st.session_state.owner,

                        st.session_state.repo,

                        st.session_state.branch,
                    )
                )


                st.session_state.thread_id = (
                    thread_id
                )


                add_log(
                    "Thread ID regenerated"
                )


            add_log(
                "Using LangGraph thread: "
                f"{thread_id}"
            )


            # ------------------------------------------------
            # INVOKE
            # ------------------------------------------------

            result = (
                st.session_state.graph.invoke(

                    {

                        "question":
                            question,

                        "owner":
                            st.session_state.owner,

                        "repo":
                            st.session_state.repo,

                        "branch":
                            st.session_state.branch,
                    },

                    config={

                        "configurable": {

                            "thread_id":
                                thread_id,
                        }
                    },
                )
            )


            duration = (
                time.perf_counter()
                - start_time
            )


            # ------------------------------------------------
            # ANSWER
            # ------------------------------------------------

            answer = result.get(
                "answer",
                "",
            )


            if not answer:

                answer = (
                    "No answer was returned "
                    "by the reasoning graph."
                )


            # ------------------------------------------------
            # DOCUMENTS
            # ------------------------------------------------

            documents = result.get(
                "documents",
                [],
            )


            related_documents = (
                result.get(
                    "related_documents",
                    [],
                )
            )


            all_documents = (
                documents
                + related_documents
            )


            add_log(
                "Primary documents: "
                f"{len(documents)}"
            )


            add_log(
                "Related documents: "
                f"{len(related_documents)}"
            )


            # ------------------------------------------------
            # SOURCES
            # ------------------------------------------------

            sources = []

            seen = set()


            for document in all_documents:

                metadata = (
                    getattr(
                        document,
                        "metadata",
                        {},
                    )
                    or {}
                )


                file_path = (
                    metadata.get(
                        "file_path",
                        "Unknown",
                    )
                )


                symbol_name = (
                    metadata.get(
                        "symbol_name",
                        "",
                    )
                )


                start_line = (
                    metadata.get(
                        "start_line",
                        "?",
                    )
                )


                end_line = (
                    metadata.get(
                        "end_line",
                        "?",
                    )
                )


                source_type = (
                    metadata.get(
                        "relationship_source",
                        "semantic",
                    )
                )


                key = (

                    file_path,

                    symbol_name,

                    start_line,

                    end_line,
                )


                if key in seen:

                    continue


                seen.add(
                    key
                )


                sources.append(

                    {

                        "file_path":
                            file_path,

                        "symbol_name":
                            symbol_name,

                        "start_line":
                            start_line,

                        "end_line":
                            end_line,

                        "source_type":
                            source_type,
                    }
                )


            # ------------------------------------------------
            # INTENT
            # ------------------------------------------------

            intent = result.get(
                "intent",
                "unknown",
            )


            # ------------------------------------------------
            # SAVE ASSISTANT MESSAGE
            # ------------------------------------------------

            st.session_state.messages.append(

                {

                    "role":
                        "assistant",

                    "content":
                        answer,

                    "sources":
                        sources,

                    "intent":
                        intent,

                    "duration":
                        duration,
                }
            )


            add_log(
                "Answer generated in "
                f"{duration:.2f}s"
            )


            add_log(
                f"Intent: {intent}"
            )


            st.rerun()


        except Exception as exc:

            duration = (
                time.perf_counter()
                - start_time
            )


            add_log(
                "GRAPH ERROR after "
                f"{duration:.2f}s: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )


            st.error(

                "Graph execution failed: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )


            with st.expander(
                "▼ DEBUG TRACE"
            ):

                st.code(

                    traceback.format_exc(),

                    language="text",
                )


# ============================================================
# SYSTEM LOG
# ============================================================

with st.expander(
    "▼ SYSTEM LOG"
):

    if st.session_state.logs:

        st.code(

            "\n".join(
                st.session_state.logs
            ),

            language="text",
        )

    else:

        st.caption(
            "No system events yet."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    'REPOSENSE // AI CODE INTELLIGENCE ENGINE'
    '<br>'
    'RAG · LANGGRAPH · CODE GRAPH · '
    'HUGGING FACE · CHROMA · GROQ'
    '<br>'
    'DYNAMIC GITHUB REPOSITORY ANALYSIS'
    '</div>',
    unsafe_allow_html=True,
)