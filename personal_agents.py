from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv(verbose=True)

model = init_chat_model(
    "gpt-4o-mini",
    temperature=0.7,
    timeout=30,
    max_tokens=1000,
)


@tool
def create_calendar_event(
        title: str,
        start_time: str,
        end_time: str,
        attendees: list[str],
        location: str,
) -> str:
    """Create a calendar event. Requires exact ISO datetime format 2024-01-01T10:00:00."""
    return f"Event created: '{title}' from {start_time} to {end_time} with {len(attendees)} attendees."

@tool
def send_email(
        to: list[str],
        subject: str,
        body: str,
        cc: list[str],
) -> str:
    """Send an email via email API. Requires properly formatted address."""
    return f"Email sent to {', '.join(to)} - Subject: {subject}."

@tool
def get_avaliable_time_slots(
        attendees: list[str],
        date: str,
        duration_minutes: int,
) -> list[str]:
    """Get calender availability for given attendees on a specific date."""
    return ["09:00","14:00","16:00"]

CALAENDER_AGENT_PROMPT = (
    "You are a calendar scheduling assistant. "
    "Parse natural language scheduling requests (e.g., 'next Tuesday at 2pm') "
    "into proper ISO datetime formats. "
    "Use get_available_time_slots to check availability when needed. "
    "Use create_calendar_event to schedule events. "
    "Always confirm what was scheduled in your final response."
)

calendar_agent = create_agent(
    model=model,
    tools=[create_calendar_event, get_avaliable_time_slots],
    system_prompt=CALAENDER_AGENT_PROMPT,
)

query = "Schedule a team meeting next Tuesday at 2pm for 1 hour"

# for step in calendar_agent.stream(
#         {"messages": [{"role": "user", "content": query}]}
# ):
#     for update in step.values():
#         for message in update.get("messages", []):
#             message.pretty_print()

EMAIL_AGENT_PROMPT = (
    "You are an email assistant. "
    "Compose professional emails based on natural language requests. "
    "Extract recipient information and craft appropriate subject lines and body text. "
    "Use send_email to send the message. "
    "Always confirm what was sent in your final response."
)
email_agent = create_agent(
    model=model,
    tools=[send_email],
    system_prompt=EMAIL_AGENT_PROMPT,
)

@tool
def schedule_event(request: str) -> str:
    """Schedule calendar events using natural language.

        Use this when the user wants to create, modify, or check calendar appointments.
        Handles date/time parsing, availability checking, and event creation.

        Input: Natural language scheduling request (e.g., 'meeting with design team
        next Tuesday at 2pm')
        """
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text

@tool
def manage_email(request: str) -> str:
    """Send emails using natural language.

    Use this when the user wants to send notifications, reminders, or any email
    communication. Handles recipient extraction, subject generation, and email
    composition.

    Input: Natural language email request (e.g., 'send them a reminder about
    the meeting')
    """
    result = email_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text

SUPERVISOR_PROMPT = (
    "You are a helpful personal assistant. "
    "You can schedule calendar events and send emails. "
    "Break down user requests into appropriate tool calls and coordinate the results. "
    "When a request involves multiple actions, use multiple tools in sequence."
)

supervisor_agent = create_agent(
    model=model,
    tools=[schedule_event, manage_email],
    system_prompt=SUPERVISOR_PROMPT,
)

query = "Schedule a team standup for tomorrow at 9am,team members are jing,jin, and send them an email reminder about the meeting."

for step in supervisor_agent.stream(
        {"messages": [{"role": "user", "content": query}]}
):
    for update in step.values():
        for message in update.get("messages", []):
            message.pretty_print()