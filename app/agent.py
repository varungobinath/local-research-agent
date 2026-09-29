import os

from dotenv import load_dotenv

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models.lite_llm import LiteLlm

from app.tools.document_search import search_documents


load_dotenv()


MODEL = os.getenv(
    "LM_STUDIO_LLM_MODEL"
)

BASE_URL = os.getenv(
    "LM_STUDIO_BASE_URL"
)

API_KEY = os.getenv(
    "LM_STUDIO_API_KEY"
)


local_llm = LiteLlm(
    model=f"openai/{MODEL}",
    api_base=BASE_URL,
    api_key=API_KEY,
)


root_agent = Agent(
    name="research_agent",

    model=local_llm,

    description=(
        "A local document research agent "
        "that answers questions using "
        "uploaded documents."
    ),

    instruction="""
You are a Local Document Research Agent.

Your job is to answer questions using
the user's local documents.

Rules:

1. If the question requires information
   from the documents, ALWAYS call
   search_documents.

2. Use the retrieved document content
   to answer the question.

3. Do not invent information.

4. If the documents do not contain
   enough information, clearly say so.

5. Mention the source document when
   possible.

6. For casual conversation that does
   not require document information,
   you can answer normally.
""",

    tools=[
        search_documents
    ],
)


app = App(
    name="app",
    root_agent=root_agent,
)