from langgraph.graph import MessagesState, START, END, StateGraph
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from opik import opik_context
from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.prebuilt import tools_condition
from typing import Literal
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader

FILE_PATH_RODI = "./.data/about_rodi.pdf"

response_model = init_chat_model("gpt-4o", temperature=0)
grader_model = init_chat_model("gpt-4o", temperature=0)

print(f"FILE_PATH_RODI: {FILE_PATH_RODI}")
docs = []
def load_sources(state: MessagesState):
  try:
    loader = PyPDFLoader(
      file_path=FILE_PATH_RODI,
      mode="page"
    )
    docs_lazy = loader.lazy_load()
    for doc in docs_lazy:
      docs.append(doc)
      print(f"docs - {docs}")
  except Exception as e:
    print(f"Error: loading PDF Failed {e}")
    
  return { "sources": True }

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=100
)
doc_splits = text_splitter.split_documents(docs)

vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits, embedding=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever()

# retriever tool
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Rohit Diwakar.",
)

#FIXME: move to src/nodes on refactor
def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = (
        response_model
        .bind_tools([retriever_tool]).invoke(state["messages"])  
    )
    return {"messages": [response]}
  
def retrieve(state: MessagesState):
    """Retrieve documents using the retriever tool."""
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    tool_messages = []
    for tool_call in tool_calls:
        result = retriever_tool.invoke(tool_call["args"])
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
    return {"messages": tool_messages}

def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    
    #TODO: move to evals fn for readability
    trace = opik_context.get_current_trace_data()
    if trace is not None:
        try:
            print(f"trace? {trace}")
            #TODO: traces, feedback scores in the end
            # feedback_scores = eval_llm(question, response)
            # opik_context.update_current_trace(feedback_scores=feedback_scores)
            print(f"trace updated")
        except Exception as e:
            print(f'Feedback Score failed: {e}') 

    return {"messages": [response]}

def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}

## GRADING

from pydantic import BaseModel, Field

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)


class GradeDocuments(BaseModel):  
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


grader_model = init_chat_model("gpt-4o", temperature=0)

def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (
        grader_model
        .with_structured_output(GradeDocuments).invoke(  
            [{"role": "user", "content": prompt}]
        )
    )
    score = response.binary_score

    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"

def create_workflow_graph():
  graph_builder = StateGraph(MessagesState)

  # Define the nodes we will cycle between
  graph_builder.add_node(load_sources)
  graph_builder.add_node(generate_query_or_respond)
  graph_builder.add_node(retrieve)
  graph_builder.add_node(rewrite_question)
  graph_builder.add_node(generate_answer)

  graph_builder.add_edge(START, "load_sources")
  graph_builder.add_edge("load_sources", "generate_query_or_respond")

  # Decide whether to retrieve
  graph_builder.add_conditional_edges(
      "generate_query_or_respond",
      # Assess LLM decision (call `retriever_tool` tool or respond to the user)
      tools_condition,
      {
          # Translate the condition outputs to nodes in our graph
          "tools": "retrieve",
          END: END,
      },
  )

  # Edges taken after the `action` node is called.
  graph_builder.add_conditional_edges(
      "retrieve",
      # Assess agent decision
      grade_documents,
  )
  graph_builder.add_edge("generate_answer", END)
  graph_builder.add_edge("rewrite_question", "generate_query_or_respond")

  return graph_builder

graph_builder = create_workflow_graph().compile()
