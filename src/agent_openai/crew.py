from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.llm import LLM
from typing import List
import os
from dotenv import load_dotenv
from agent_openai.tools.mongodb_tool import MongoDBTool

load_dotenv()

@CrewBase
class AgentOpenai():
    agents: List[BaseAgent]
    tasks: List[Task]
    @agent
    def validation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['validation_agent'], 
            tools=[MongoDBTool()],
            llm=LLM(
                model=os.getenv("MODEL", "o4-mini"),
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.3,
                max_tokens=4000
            ),
            verbose=True,
            system_template="""You are a validation agent. You MUST use the available tools to get real data. 
        NEVER fabricate or make up database records. Always call the MongoDBTool first before providing any analysis."""
    )
    @task
    def validation_task(self) -> Task:
        return Task(
            config=self.tasks_config['validation_task'],
            description="""
            Step 1: Extract user ID from this message: {topic}
            Step 2: Use MongoDBTool() to query database with the user ID
            Step 3: Report the exact database response
            Step 4: Analyze only the real database data
            
            DO NOT PROCEED WITHOUT CALLING MongoDBTool FIRST.
            """,
            agent=self.validation_agent(),
            output_file='user_report.md' # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
