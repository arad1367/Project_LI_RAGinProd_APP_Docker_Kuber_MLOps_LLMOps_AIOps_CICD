import os
import sys
import ast
import operator

from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from retrieve import retrieve


# ============================================================
# 1) TOOLS — plain Python functions the agent can choose to call.
#    The docstring is what the model reads to decide WHEN to use each one,
#    so write it clearly.
# ============================================================

@tool
def search_documents(query: str) -> str:
    """Search the user's ingested documents for information relevant to a query.
    Use this whenever the question might be answered by the documents."""
    hits = retrieve(query, k=5)
    if not hits:
        return "No relevant documents found."
    blocks = []
    for h in hits:
        loc = h["source"] + (f" p{h['page']}" if h["page"] else "")
        if h["type"] == "image":
            blocks.append(f"[{loc}] (an image matched this query)")
        else:
            blocks.append(f"[{loc}]\n{h['text']}")
    return "\n\n".join(blocks)


# A tiny safe arithmetic evaluator (no eval() — only + - * / % ** and parens).
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg,
}

def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")

@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression like '2 + 2 * 5'.
    Use this for math instead of computing it in your head."""
    try:
        return str(_safe_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Could not evaluate '{expression}': {e}"


TOOLS = [search_documents, calculator]


# ============================================================
# 2) THE BRAIN — a chat model that supports tool calling.
#    Swappable: defaults to your working gpt-4o-mini.
# ============================================================

def get_model(provider="openai"):
    if provider == "openai":
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            google_api_key=os.environ["GEMINI_API_KEY"],
            temperature=0,
        )
    if provider == "local":
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3.1:8b", temperature=0)
    raise ValueError(f"Unknown provider: {provider}")


SYSTEM = SystemMessage(content=(
    "You are DocuMind, a helpful assistant. You have two tools: "
    "search_documents (to look things up in the user's documents) and "
    "calculator (for arithmetic). Decide for yourself when a tool is needed. "
    "When you use search_documents, answer ONLY from what it returns and cite "
    "the source filenames. If the documents don't contain the answer, say so."
))


# ============================================================
# 3) THE GRAPH — wire the brain and tools into a thinking loop.
# ============================================================

def build_agent(provider="openai"):
    model_with_tools = get_model(provider).bind_tools(TOOLS)

    # The assistant node: prepend the system message, then let the model decide.
    def assistant(state: MessagesState):
        return {"messages": [model_with_tools.invoke([SYSTEM] + state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "assistant")
    # Conditional edge: if the assistant asked for a tool -> "tools", else -> END.
    builder.add_conditional_edges("assistant", tools_condition)
    # After tools run, go back to the assistant to think again. This is THE LOOP.
    builder.add_edge("tools", "assistant")

    # checkpointer = memory across turns within a conversation thread.
    return builder.compile(checkpointer=MemorySaver())


# ============================================================
# 4) RUN IT — an interactive loop so follow-up questions have memory.
# ============================================================

if __name__ == "__main__":
    provider = "openai"
    if len(sys.argv) > 1 and sys.argv[1] in ("openai", "gemini", "local"):
        provider = sys.argv[1]

    agent = build_agent(provider)
    print(agent.get_graph().draw_ascii()) # Show actual graph
    # The thread_id ties messages together so the agent remembers the conversation.
    config = {"configurable": {"thread_id": "session-1"}}

    print(f"DocuMind agent ready (brain: {provider}). Ask a question, or type 'quit'.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit", ""):
            break
        # stream_mode="values" yields the full message list after each step,
        # so you can WATCH the agent think: tool call -> tool result -> answer.
        for chunk in agent.stream(
            {"messages": [HumanMessage(content=q)]},
            config,
            stream_mode="values",
        ):
            chunk["messages"][-1].pretty_print()
        print()