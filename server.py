from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
from emergentintegrations.llm.chat import LlmChat, UserMessage
import asyncio
import json

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Models
class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    email: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserCreate(BaseModel):
    username: str
    email: str

class PlanningSession(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    title: str
    status: str = 'active'  # 'active', 'completed', 'archived'
    conversation: List[Dict[str, Any]] = []
    generated_prompt: Optional[str] = None
    final_response: Optional[str] = None
    ai_context: Optional[str] = None  # AI's understanding of the planning context
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class SessionCreate(BaseModel):
    user_id: str
    title: str
    initial_prompt: str

class ChatMessage(BaseModel):
    session_id: str
    message: str
    is_user: bool = True

class PromptApproval(BaseModel):
    session_id: str
    approved_prompt: str

# Helper functions
def prepare_for_mongo(data):
    """Convert datetime objects to ISO strings for MongoDB storage"""
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, list):
                data[key] = [prepare_for_mongo(item) if isinstance(item, dict) else item for item in value]
    return data

def parse_from_mongo(item):
    """Convert ISO strings back to datetime objects"""
    if isinstance(item, dict):
        for key, value in item.items():
            if key in ['created_at', 'updated_at'] and isinstance(value, str):
                item[key] = datetime.fromisoformat(value)
    return item

async def get_ai_chat():
    """Initialize AI chat with credentials"""
    gemini_key = os.environ.get('GEMINI_API_KEY')
    emergent_key = os.environ.get('EMERGENT_LLM_KEY')
    
    api_key = gemini_key if gemini_key else emergent_key
    
    chat = LlmChat(
        api_key=api_key,
        session_id=f"leo_planning_assistant",
        system_message="You are Leo, an intelligent AI planning assistant. You help users create comprehensive, actionable plans by asking thoughtful, contextual questions. You understand the user's goal from their initial message and ask relevant follow-up questions to gather all necessary information. When you have sufficient information, you indicate completion and help generate a detailed plan. Try gathering the information in 10 prompts to keep the user engaged."
    ).with_model("gemini", "gemini-2.0-flash")
    
    return chat

async def analyze_initial_prompt(initial_prompt: str) -> Dict[str, Any]:
    """Use AI to analyze the initial prompt and determine the planning context"""
    try:
        chat = await get_ai_chat()
        
        analysis_prompt = f"""
Analyze this planning request and provide a structured response:

User's Request: "{initial_prompt}"

Please respond with a JSON object containing:
{{
    "planning_type": "brief category (e.g., business, personal, creative, technical)",
    "main_goal": "the primary objective",
    "context_summary": "brief understanding of what they want to achieve",
    "first_question": "an intelligent, contextual follow-up question to gather more specific information"
}}

Make sure your response is valid JSON only, no additional text.
"""
        
        response = await chat.send_message(UserMessage(text=analysis_prompt))
        
        # Parse the JSON response
        try:
            analysis = json.loads(response)
            return analysis
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "planning_type": "general",
                "main_goal": initial_prompt,
                "context_summary": f"User wants to plan: {initial_prompt}",
                "first_question": "Can you tell me more about your specific goals and what you hope to achieve?"
            }
    except Exception as e:
        logging.error(f"Error analyzing initial prompt: {str(e)}")
        return {
            "planning_type": "general",
            "main_goal": initial_prompt,
            "context_summary": f"User wants to plan: {initial_prompt}",
            "first_question": "Can you tell me more about your specific goals and what you hope to achieve?"
        }

async def generate_next_question(session_id: str, conversation_history: List[Dict]) -> Dict[str, Any]:
    """Use AI to generate the next contextual question or determine if ready for final plan"""
    try:
        chat = await get_ai_chat()
        
        # Build conversation context
        conversation_text = "\n".join([
            f"{'User' if msg['is_user'] else 'Leo'}: {msg['content']}"
            for msg in conversation_history[-10:]  # Last 10 messages for context
        ])
        
        decision_prompt = f"""
Based on this conversation with a user who wants help with planning, decide what to do next:

Conversation so far:
{conversation_text}

Analyze the conversation and respond with a JSON object:
{{
    "action": "continue" or "complete",
    "reasoning": "brief explanation of your decision",
    "next_question": "if continuing, what specific question should I ask next?",
    "readiness_score": "1-10 scale of how ready we are to create a comprehensive plan"
}}

Guidelines:
- Choose "continue" if you need more specific information for a comprehensive plan
- Choose "complete" if you have enough information to create a detailed, actionable plan
- Ask contextual questions that build on previous answers
- Focus on gathering information about goals, resources, timeline, challenges, and success metrics

Respond with valid JSON only, no additional text.
"""
        
        response = await chat.send_message(UserMessage(text=decision_prompt))
        
        try:
            decision = json.loads(response)
            return decision
        except json.JSONDecodeError:
            # Fallback - continue with generic question
            return {
                "action": "continue",
                "reasoning": "Need more information",
                "next_question": "What other important details should I know to help create the best plan for you?",
                "readiness_score": "5"
            }
    except Exception as e:
        logging.error(f"Error generating next question: {str(e)}")
        return {
            "action": "continue",
            "reasoning": "Need more information",
            "next_question": "Can you provide more details to help me create a comprehensive plan?",
            "readiness_score": "4"
        }

async def generate_comprehensive_prompt(session_id: str, conversation_history: List[Dict]) -> str:
    """Generate comprehensive planning prompt based on entire conversation"""
    try:
        chat = await get_ai_chat()
        
        # Build full conversation context
        conversation_text = "\n".join([
            f"{'User' if msg['is_user'] else 'Leo'}: {msg['content']}"
            for msg in conversation_history
        ])
        
        prompt_generation = f"""
Based on this entire conversation, create a comprehensive planning prompt that will generate the best possible personalized plan:

Full Conversation:
{conversation_text}

Create a detailed prompt that includes:
- Clear summary of the user's goals and objectives
- All important context and constraints they've provided
- Specific request for actionable steps and timeline
- Request for potential challenges and solutions
- Request for success metrics and milestones
- Request for resource requirements
- Request for next immediate actions

Make the prompt comprehensive and specific so it will generate an excellent, personalized, actionable plan.
"""
        
        response = await chat.send_message(UserMessage(text=prompt_generation))
        return response
        
    except Exception as e:
        logging.error(f"Error generating comprehensive prompt: {str(e)}")
        # Fallback prompt
        return f"Please create a comprehensive, actionable plan based on our conversation. Include specific steps, timeline, resources needed, potential challenges and solutions, and success metrics."

# API Routes
@api_router.post("/users", response_model=User)
async def create_user(user_data: UserCreate):
    user_obj = User(**user_data.model_dump())
    doc = prepare_for_mongo(user_obj.model_dump())
    await db.users.insert_one(doc)
    return user_obj

@api_router.get("/users/{user_id}", response_model=User)
async def get_user(user_id: str):
    user = await db.users.find_one({"id": user_id}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return parse_from_mongo(user)

@api_router.post("/sessions", response_model=PlanningSession)
async def create_session(session_data: SessionCreate):
    # Verify user exists
    user = await db.users.find_one({"id": session_data.user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Analyze the initial prompt with AI
    analysis = await analyze_initial_prompt(session_data.initial_prompt)
    
    session_obj = PlanningSession(
        user_id=session_data.user_id,
        title=session_data.title,
        ai_context=json.dumps(analysis)
    )
    
    # Add initial greeting and first AI-generated question
    initial_message = {
        "content": f"Hello! I'm Leo, your AI planning assistant. I understand you want to {analysis['context_summary'].lower()}. I'm here to help you create a comprehensive, personalized plan by asking you the right questions.\n\nLet's start:",
        "is_user": False,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    first_question = {
        "content": analysis['first_question'],
        "is_user": False,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    session_obj.conversation = [initial_message, first_question]
    
    doc = prepare_for_mongo(session_obj.model_dump())
    await db.planning_sessions.insert_one(doc)
    return session_obj

@api_router.get("/sessions/user/{user_id}", response_model=List[PlanningSession])
async def get_user_sessions(user_id: str):
    sessions = await db.planning_sessions.find({"user_id": user_id}, {"_id": 0}).to_list(1000)
    return [parse_from_mongo(session) for session in sessions]

@api_router.get("/sessions/{session_id}", response_model=PlanningSession)
async def get_session(session_id: str):
    session = await db.planning_sessions.find_one({"id": session_id}, {"_id": 0})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return parse_from_mongo(session)

@api_router.post("/chat")
async def send_message(chat_data: ChatMessage):
    session = await db.planning_sessions.find_one({"id": chat_data.session_id})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Add user message to conversation
    user_message = {
        "content": chat_data.message,
        "is_user": True,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    conversation = session.get('conversation', [])
    conversation.append(user_message)
    
    # Use AI to determine next action
    decision = await generate_next_question(chat_data.session_id, conversation)
    
    if decision['action'] == 'continue':
        # Generate next question
        leo_response = {
            "content": decision['next_question'],
            "is_user": False,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        conversation.append(leo_response)
        
        # Update session
        await db.planning_sessions.update_one(
            {"id": chat_data.session_id},
            {"$set": {
                "conversation": conversation,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }}
        )
        
        return {
            "message": leo_response["content"],
            "completed": False,
            "readiness_score": decision.get('readiness_score', '5')
        }
    else:
        # Ready to generate comprehensive plan
        completion_message = {
            "content": f"Perfect! I have gathered comprehensive information about your planning needs. Based on our conversation, I now have a clear understanding of your goals, constraints, and requirements.\n\nI'm ready to create a detailed, personalized plan for you. Let me prepare a comprehensive prompt that will generate the best possible planning advice.",
            "is_user": False,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        conversation.append(completion_message)
        
        # Generate the comprehensive prompt using AI
        generated_prompt = await generate_comprehensive_prompt(chat_data.session_id, conversation)
        
        # Update session
        await db.planning_sessions.update_one(
            {"id": chat_data.session_id},
            {"$set": {
                "conversation": conversation,
                "generated_prompt": generated_prompt,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }}
        )
        
        return {
            "message": completion_message["content"],
            "completed": True,
            "generated_prompt": generated_prompt,
            "readiness_score": decision.get('readiness_score', '10')
        }

@api_router.post("/approve-prompt")
async def approve_prompt(approval_data: PromptApproval):
    session = await db.planning_sessions.find_one({"id": approval_data.session_id})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        chat = await get_ai_chat()
        
        # Send the approved prompt to generate final plan
        user_message = UserMessage(text=approval_data.approved_prompt)
        response = await chat.send_message(user_message)
        
        # Add Leo's final response to conversation
        leo_final_response = {
            "content": response,
            "is_user": False,
            "is_final_response": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        conversation = session.get('conversation', [])
        conversation.append(leo_final_response)
        
        # Update session with final response
        await db.planning_sessions.update_one(
            {"id": approval_data.session_id},
            {"$set": {
                "conversation": conversation,
                "final_response": response,
                "status": "completed",
                "updated_at": datetime.now(timezone.utc).isoformat()
            }}
        )
        
        return {"response": response, "success": True}
    
    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

@api_router.get("/")
async def root():
    return {"message": "Leo AI Planning Assistant API"}