from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo 

from dotenv import load_dotenv
load_dotenv()

websearch_agent = Agent(
    name="web Search Agent",
    role="Search the web for information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tools_calls=True,
    markdown=True
)

finanace_agent = Agent(
    name="Finance AI Agent",
    role="Search the web for information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    show_tool_calls=True,
    instructions=["Use tables to display data"],
)

team_agent= Agent(
    team=[finanace_agent, websearch_agent],
    show_tool_calls=True,
    instructions=["Always include sources","Use tables to display data"],
    markdown=True,
    model=Groq(id="llama-3.1-70b-versatile")
)

team_agent.print_response("summarize analyst recommendations for and share latest news for NVDA",stream=True)