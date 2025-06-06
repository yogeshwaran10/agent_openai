import os
import json
from typing import Type
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

load_dotenv()

class MongoDBToolInput(BaseModel):
    user_id: str = Field(..., description="The user ID to query in the database.")

class MongoDBTool(BaseTool):
    name: str = "MongoDB User Query Tool"
    description: str = (
        "Use this tool to retrieve user records from the MongoDB database using their user ID. "
        "This tool returns the EXACT database record as it exists in the database. "
        "IMPORTANT: DO NOT MODIFY THE DATABASE RECORD IN ANY WAY. Use the record EXACTLY as returned - "
        "do not rename fields, do not create new fields, do not modify values, do not reformat the data. "
        "The exact record and its structure must be preserved and displayed as is."
    )
    args_schema: Type[BaseModel] = MongoDBToolInput

    def _connect_to_mongodb(self, mongodb_uri):
        try:
            client = MongoClient(mongodb_uri)
            client.admin.command('ping')
            return client, client.crewai
        except ConnectionFailure as e:
            raise ConnectionFailure(f"Failed to connect to MongoDB: {str(e)}")
        except Exception as e:
            raise Exception(f"An error occurred while connecting to MongoDB: {str(e)}")

    def _run(self, user_id: str) -> str:
        mongodb_uri = os.getenv("MONGODB_URI")
        if not mongodb_uri:
            return json.dumps({"error": "MongoDB URI not found in environment variables. Please set MONGODB_URI in your .env file."})
        
        client = None
        try:
            client, db = self._connect_to_mongodb(mongodb_uri)
            users_collection = db["testing"]
            user = users_collection.find_one({"userid": user_id})
            
            if user:
                if "_id" in user:
                    user["_id"] = str(user["_id"])
                result = json.dumps(user, default=str)
                return result
            else:
                print(f"\n\nUser with ID {user_id} not found in MongoDB\n\n")
                return json.dumps({"error": f"User with ID {user_id} not found"})
        except ConnectionFailure as e:
            return json.dumps({"error": f"Failed to connect to MongoDB: {str(e)}"})
        except OperationFailure as e:
            return json.dumps({"error": f"MongoDB operation failed: {str(e)}"})
        except Exception as e:
            return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})
        finally:
            if client:
                client.close()
