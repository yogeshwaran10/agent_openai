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
    def freshdesk_support_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['freshdesk_support_agent'], 
            tools=[MongoDBTool()],
            llm=LLM(
                model=os.getenv("MODEL", "o4-mini-2025-04-16"),
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.1,  # Lower temperature for more consistent behavior
                max_tokens=4000
            ),
            function_calling_llm=LLM(
                model="o4-mini-2025-04-16",
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.1
            ),
            verbose=True,
            max_retry_limit=3,
            system_template="""You are a Freshdesk Support Agent who MUST use tools to resolve customer queries.

CRITICAL INSTRUCTIONS:
1. You MUST call the MongoDBTool() for EVERY user query that contains a user ID
2. NEVER provide solutions without first fetching the user's database record
3. NEVER fabricate or assume database information
4. If you cannot find a user ID in the query, ask for it before proceeding
5. Always use the exact database record returned by the tool

MANDATORY WORKFLOW:
Step 1: Extract user ID from the customer message
Step 2: Call MongoDBTool() with the user ID (THIS IS MANDATORY)
Step 3: Analyze the returned database record
Step 4: Provide solution based on ACTUAL database data only

You can use the tool: MongoDBTool() for fetching customer details from MongoDB to resolve customer queries."""
        )
    
    @agent  
    def validation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config.get('validation_agent', self.agents_config['freshdesk_support_agent']),
            tools=[MongoDBTool()],
            llm=LLM(
                model=os.getenv("MODEL", "o4-mini-2025-04-16"),
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.1,
                max_tokens=4000
            ),
            function_calling_llm=LLM(
                model="gpt-4o",
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.1
            ),
            verbose=True,
            max_retry_limit=3,
            system_template="""You are a database validation agent who MUST use the MongoDBTool.

CRITICAL: You MUST call MongoDBTool() for every task. NO EXCEPTIONS.
NEVER proceed without calling the tool first.
NEVER fabricate database records.
Always use the exact data returned by the tool."""
        )

    @task
    def validation_task(self) -> Task:
        return Task(
            config=self.tasks_config['validation_task'],
            description="""
            MANDATORY STEPS - FOLLOW IN EXACT ORDER:
            
            Step 1: Extract user ID from this message: {topic}
            Step 2: CALL MongoDBTool() with the extracted user ID (THIS IS REQUIRED - DO NOT SKIP)
            Step 3: Wait for and analyze the EXACT database response 
            Step 4: Provide solution based ONLY on the retrieved database data
            
            CRITICAL: DO NOT PROCEED WITHOUT CALLING MongoDBTool FIRST.
            DO NOT FABRICATE ANY DATABASE INFORMATION.
            USE ONLY THE EXACT DATA RETURNED BY THE TOOL.
            """,
            agent=self.validation_agent(),
            tools=[MongoDBTool()],
            expected_output="""
            A comprehensive user report that includes:
            1. Extracted user information (ID, region, etc.)
            2. EXACT database record retrieved using MongoDBTool()
            3. Analysis of the database record
            4. Solution based on the actual database data
            
            MUST include proof that MongoDBTool() was called and its exact results.
            """,
            output_file='user_report.md'
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True,
        )
