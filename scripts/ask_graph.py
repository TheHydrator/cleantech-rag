import sys
from dotenv import load_dotenv

from rag.graph import build_graph

load_dotenv()

args = sys.argv[1:]
use_hyde = "--hyde" in args
args = [a for a in args if a != "--hyde"]
question = " ".join(args) or "What is the EU's Green Deal Industrial Plan?"

app = build_graph()
state = app.invoke({"query": question}, config={"configurable": {"use_hyde": use_hyde}})

print(f"Question: {state['query']}  (HyDE: {'on' if state['hyde_used'] else 'off'})\n")
if state.get("hypothesis"):
    print(f"Hypothesis:\n{state['hypothesis']}\n")
if state.get("hyde_error"):
    print(f"HyDE failed, fell back to normal search: {state['hyde_error']}\n")
print(f"Answer:\n{state['final_answer']}\n")
print(f"Grade: {state['grade']}  ({state.get('grade_reasoning', '')})\n")
print("Sources:", ", ".join(f"{r[0]} ({r[4]:.2f})" for r in state["retrieved"]))
print(f"Tokens: {state['usage']['in']} in, {state['usage']['out']} out")