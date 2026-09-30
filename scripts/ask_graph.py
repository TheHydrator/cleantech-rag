import sys
from dotenv import load_dotenv

from rag.graph import build_graph

load_dotenv()
question = " ".join(sys.argv[1:]) or "What is the EU's Green Deal Industrial Plan?"

app = build_graph()
state = app.invoke({"query": question})

print(f"Question: {state['query']}\n")
print(f"Answer:\n{state['final_answer']}\n")
print("Sources:", ", ".join(f"{r[0]} ({r[4]:.2f})" for r in state["retrieved"]))
print(f"Tokens: {state['usage']['in']} in, {state['usage']['out']} out")