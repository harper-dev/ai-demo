from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langchain.agents import AgentState, create_agent
from typing_extensions import NotRequired
from typing import Literal
from langchain.messages import ToolMessage, HumanMessage
from langgraph.types import Command
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable
from langgraph.checkpoint.memory import InMemorySaver
import uuid


load_dotenv(verbose=True)
model = init_chat_model(
    "gpt-4o-mini",
    temperature=0.7,
    timeout=30,
    max_tokens=1000,
)

SupportStep = Literal["warranty_collector","issue_classifier","resolution_specialist"]
class SupportState(AgentState):
    """Custom runtime context schema."""
    current_step: NotRequired[SupportStep]
    warranty_status: NotRequired[Literal["in_warranty","out_of_warranty"]]
    issue_type: NotRequired[Literal["hardware","software","network"]]

@tool
def record_warranty_status(
        status: Literal["in_warranty","out_of_warranty"],
        runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record the customer's warranty status and transition to issue classification."""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Warranty status recorded as {status}. Proceeding to issue classification.",
                    tool_call_id=runtime.tool_call_id
                )
            ],
            "warranty_status": status,
            "current_step": "issue_classifier",
        }
    )

@tool
def record_issue_type(
        issue_type: Literal["hardware","software"],
        runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record the type of issue and transition to resolution specialist."""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Issue type recorded as {issue_type}",
                    tool_call_id=runtime.tool_call_id
                )
            ],
            "issue_type": issue_type,
            "current_step": "resolution_specialist",
        }
    )

@tool
def escalate_to_human(reason: str) -> str:
    """Escalate the case to a human support agent."""
    return f"Escalating to human support. Reason: {reason}."

@tool
def provide_solution(solution: str) -> str:
    """Provide a solution to the customer's issue."""
    return f"Solution provided to customer: {solution}."

@tool
def go_back_to_warranty() -> Command:
    """Go back to warranty verification step."""
    return Command(
        update={
            "current_step": "warranty_collector",
        }
    )
@tool
def go_back_to_classification() -> Command:
    """Go back to issue classification step."""
    return Command(
        update={
            "current_step": "issue_classifier",
        }
    )

# Define prompts as constants for easy reference
WARRANTY_COLLECTOR_PROMPT = """You are a customer support agent helping with device issues.

CURRENT STAGE: Warranty verification

At this step, you need to:
1. Greet the customer warmly
2. Ask if their device is under warranty
3. Use record_warranty_status to record their response and move to the next step

Be conversational and friendly. Don't ask multiple questions at once."""

ISSUE_CLASSIFIER_PROMPT = """You are a customer support agent helping with device issues.

CURRENT STAGE: Issue classification
CUSTOMER INFO: Warranty status is {warranty_status}

At this step, you need to:
1. Ask the customer to describe their issue
2. Determine if it's a hardware issue (physical damage, broken parts) or software issue (app crashes, performance)
3. Use record_issue_type to record the classification and move to the next step

If unclear, ask clarifying questions before classifying."""

RESOLUTION_SPECIALIST_PROMPT = """You are a customer support agent helping with device issues.

CURRENT STAGE: Resolution
CUSTOMER INFO: Warranty status is {warranty_status}, issue type is {issue_type}

At this step, you need to:
1. For SOFTWARE issues: provide troubleshooting steps using provide_solution
2. For HARDWARE issues:
   - If IN WARRANTY: explain warranty repair process using provide_solution
   - If OUT OF WARRANTY: escalate_to_human for paid repair options
If the customer indicates any information was wrong, use:
- go_back_to_warranty to correct warranty status
- go_back_to_classification to correct issue type

Be specific and helpful in your solutions."""

STEP_CONFIG={
    "warranty_collector": {
        "prompt": WARRANTY_COLLECTOR_PROMPT,
        "tools": [record_warranty_status],
        "requires": [],
    },
    "issue_classifier": {
        "prompt": ISSUE_CLASSIFIER_PROMPT,
        "tools": [record_issue_type],
        "requires": ["warranty_status"],
    },
    "resolution_specialist": {
        "prompt": RESOLUTION_SPECIALIST_PROMPT,
        "tools": [provide_solution, escalate_to_human],
        "requires": ["warranty_status", "issue_type"],
    },
}

STEP_CONFIG["resolution_specialist"]["tools"].extend([
    go_back_to_warranty,
    go_back_to_classification,
])

@wrap_model_call
def apply_step_config(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Configure agent behavior based on the current step."""
    #Get current step from state (default to warranty_collector for first interaction)
    current_step = request.state.get("current_step", "warranty_collector")
    #Look up step configuration
    state_config = STEP_CONFIG[current_step]
    #Validate required state exists
    for key in state_config["requires"]:
        if request.state.get(key) is None:
            raise ValueError(f"Missing required state '{key}' for step '{current_step}'")
    # Format prompt with state values (supports {warranty_status},    {issue_type})
    system_prompt = state_config["prompt"].format(**request.state)
    #Injedct system prompt and step-specific tools into the request
    request = request.override(
        system_prompt=system_prompt,
        tools=state_config["tools"],
    )
    return handler(request)

all_tools = [
    record_warranty_status,
    record_issue_type,
    provide_solution,
    escalate_to_human,
]
agent = create_agent(
    model=model,
    tools=all_tools,
    state_schema=SupportState,
    middleware=[apply_step_config],
    checkpointer=InMemorySaver(),
)

thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

#Turn 1: Initial message - starts with warranty_collector step
print("=== Turn 1: warranty collection ===")
result = agent.invoke(
    {"messages": [HumanMessage("Hi, my phone is cracked.")]},
    config
)
for msg in result["messages"]:
    msg.pretty_print()

#Turn 2: User responds about warranty
print("\n=== Turn 2: issue classification ===")
result = agent.invoke(
    {"messages": [HumanMessage("Yes, it's still under warranty.")]},
    config
)
for msg in result["messages"]:
    msg.pretty_print()
print(f"Current step: {result['current_step']}")

#Turn 3: User describes issue
print("\n=== Turn 3: resolution specialist ===")
result = agent.invoke(
    {"messages": [HumanMessage("The screen is physically cracked from dropping it")]},
    config
)
for msg in result["messages"]:
    msg.pretty_print()
print(f"Current  step: {result['current_step']}")

#Turn 4: Agent provides solution
print("\n=== Turn 4: resolution provided ===")
result = agent.invoke(
    {"messages": [HumanMessage("WHat should i do?")]},
    config
)
for msg in result["messages"]:
    msg.pretty_print()
