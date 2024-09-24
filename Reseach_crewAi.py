from crewai import Agent, Task
from langchain.tools import Tool
from langgraph.graph import Graph, END
from langgraph.prebuilt import ToolInvocation
from typing import TypedDict, Annotated, Sequence
import operator

# Define the state structure
class AgentState(TypedDict):
    messages: Annotated[Sequence[str], operator.add]
    current_agent: str
    research_plan: str
    research_data: list
    evaluation_results: list
    final_report: str

# Create CrewAI agents
planner = Agent(
    role="Research Planner",
    goal="Develop comprehensive research strategies",
    backstory="An experienced research strategist",
    allow_delegation=False
)

researcher = Agent(
    role="Research Specialist",
    goal="Conduct thorough RAG searches",
    backstory="A meticulous researcher",
    allow_delegation=False
)

evaluator = Agent(
    role="Research Evaluator",
    goal="Critically assess research findings",
    backstory="An analytical expert",
    allow_delegation=False
)

writer = Agent(
    role="Research Report Writer",
    goal="Synthesize findings into reports",
    backstory="A skilled technical writer",
    allow_delegation=False
)

# Define agent functions
def plan_research(state):
    task = Task(description="Develop a comprehensive research plan")
    plan = planner.execute(task)
    return {"messages": [f"Research plan: {plan}"], "research_plan": plan, "current_agent": "researcher"}

def conduct_research(state):
    task = Task(description=f"Conduct RAG searches based on the plan: {state['research_plan']}")
    data = researcher.execute(task)
    return {"messages": [f"Research data: {data}"], "research_data": data, "current_agent": "evaluator"}

def evaluate_research(state):
    task = Task(description=f"Evaluate the gathered information: {state['research_data']}")
    evaluation = evaluator.execute(task)
    return {"messages": [f"Evaluation results: {evaluation}"], "evaluation_results": evaluation, "current_agent": "writer"}

def write_report(state):
    task = Task(description=f"Compile findings into a report based on: {state['evaluation_results']}")
    report = writer.execute(task)
    return {"messages": [f"Final report: {report}"], "final_report": report, "current_agent": END}

# Create the LangGraph structure
workflow = Graph()

workflow.add_node("planner", plan_research)
workflow.add_node("researcher", conduct_research)
workflow.add_node("evaluator", evaluate_research)
workflow.add_node("writer", write_report)

workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "evaluator")
workflow.add_edge("evaluator", "writer")

# Add a feedback loop for iteration if needed
workflow.add_edge("evaluator", "researcher")

def router(state):
    return state["current_agent"]

workflow.set_entry_point("planner")
workflow.add_router(router)

# Compile the workflow
app = workflow.compile()

# Function to run the research process
def run_research_process(research_topic):
    initial_state = {
        "messages": [f"Initial research topic: {research_topic}"],
        "current_agent": "planner",
        "research_plan": "",
        "research_data": [],
        "evaluation_results": [],
        "final_report": ""
    }

    for output in app.stream(initial_state):
        if output["current_agent"] == END:
            print("Research process completed.")
            print(f"Final report: {output['final_report']}")
            return output['final_report']
        else:
            print(f"Current agent: {output['current_agent']}")
            print(f"Latest message: {output['messages'][-1]}")

# Main execution
if __name__ == "__main__":
    research_topic = "The impact of artificial intelligence on job markets"
    final_report = run_research_process(research_topic)
    print("\nResearch process completed successfully.")