import json
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from agent import build_agent
from llm import chat  # our provider abstraction, reused as the judge

agent = build_agent("openai")

# A small fixed eval set. Add 2-3 questions whose answers ARE in your docs.
EVAL_SET = [
    {"q": "what is the GPQA percent for claude opus 4.6?",
     "note": "Should answer from the documents and cite a source filename."},
    {"q": "what is 1847 * 56",
     "note": "Pure math; correct answer is 103432."},
    {"q": "who won the world cup in 1998",
     "note": "Not in the documents; it's fine to answer from general knowledge or say it's not in the docs."},
    # {"q": "what does the Pejman document say", "note": "Should summarize Pejman.txt."},
]

JUDGE_SYSTEM = (
    "You are a strict evaluator of an AI assistant's answers. "
    "Score FAITHFULNESS (is the answer grounded/correct, not made up) and "
    "RELEVANCE (does it actually answer the question), each 1-5."
)

def get_answer(question, i):
    # Unique thread per item so the agent doesn't 'remember' previous eval questions.
    state = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        {"configurable": {"thread_id": f"eval-{i}"}},
    )
    return state["messages"][-1].content

def judge(question, answer, note):
    prompt = (
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Guidance: {note}\n\n"
        "Respond ONLY with JSON, no markdown:\n"
        '{"faithfulness": <1-5>, "relevance": <1-5>, "reason": "<short>"}'
    )
    raw = chat(
        [{"role": "system", "content": JUDGE_SYSTEM},
         {"role": "user", "content": prompt}],
        provider="openai",
    )
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)

def main():
    print(f"Running {len(EVAL_SET)} eval items...\n")
    rows = []
    for i, item in enumerate(EVAL_SET):
        answer = get_answer(item["q"], i)
        try:
            scores = judge(item["q"], answer, item["note"])
        except Exception as e:
            scores = {"faithfulness": "?", "relevance": "?", "reason": f"judge error: {e}"}
        rows.append((item["q"], scores, answer))
        print(f"Q: {item['q']}")
        print(f"   faithfulness={scores['faithfulness']}  relevance={scores['relevance']}")
        print(f"   reason: {scores.get('reason','')}")
        print(f"   answer: {answer[:120]}...\n")

    # Simple aggregate
    fs = [r[1]['faithfulness'] for r in rows if isinstance(r[1]['faithfulness'], int)]
    rs = [r[1]['relevance'] for r in rows if isinstance(r[1]['relevance'], int)]
    if fs:
        print(f"AVG faithfulness: {sum(fs)/len(fs):.2f}   AVG relevance: {sum(rs)/len(rs):.2f}")

if __name__ == "__main__":
    main()