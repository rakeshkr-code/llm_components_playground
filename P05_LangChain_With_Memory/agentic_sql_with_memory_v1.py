import sqlite3
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from typing import List
import os
from dotenv import load_dotenv

# ============================================================
# MEMORY: In-Memory Chat History (FIXED)
# ============================================================

class InMemoryChatHistory(BaseChatMessageHistory):
    """Simple in-memory chat history"""
    
    def __init__(self):
        self.messages: List[BaseMessage] = []
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the history"""
        self.messages.append(message)
        print(f"[DEBUG] Added to history: {type(message).__name__}: {message.content[:50]}...")
    
    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add multiple messages to history"""
        for message in messages:
            self.add_message(message)
    
    def clear(self) -> None:
        """Clear all messages"""
        self.messages = []
        print("[DEBUG] History cleared")
    
    @property
    def messages_list(self) -> List[BaseMessage]:
        """Get all messages"""
        return self.messages

# Global conversation history
conversation_history = InMemoryChatHistory()

# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

MODEL_CONFIGS = {
    "qwen": {
        "name": "Qwen/Qwen2.5-72B-Instruct",
        "description": "Best overall - 72B params, excellent reasoning",
        "max_tokens": 4096,
    },
    "mixtral": {
        "name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "description": "Fast and good - 47B effective params",
        "max_tokens": 4096,
    },
    "llama": {
        "name": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "description": "Meta's best - 70B params",
        "max_tokens": 4096,
    },
    "command-r": {
        "name": "CohereForAI/c4ai-command-r-plus",
        "description": "Excellent for tools and RAG - 104B params",
        "max_tokens": 4096,
    },
    "phi": {
        "name": "microsoft/Phi-3-medium-4k-instruct",
        "description": "Smaller but capable - 14B params",
        "max_tokens": 4096,
    }
}

SELECTED_MODEL = "qwen"

# ============================================================
# DATABASE SETUP
# ============================================================

def create_database():
    """Create SQLite database with fruit table"""
    print("Creating database...")
    
    conn = sqlite3.connect('fruits_v2.db')
    cursor = conn.cursor()
    
    cursor.execute('DROP TABLE IF EXISTS fruit_table')
    
    cursor.execute('''
        CREATE TABLE fruit_table (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            color TEXT NOT NULL,
            price REAL,
            stock INTEGER
        )
    ''')

    fruits_data = [
        ('Apple', 'Red', 2.5, 100),
        ('Banana', 'Yellow', 1.2, 150),
        ('Kiwi', 'Green', 3.0, 80),
        ('Orange', 'Orange', 2.0, 120),
        ('Grape', 'Purple', 4.5, 60),
        ('Mango', 'Yellow', 3.5, 90),
        ('Strawberry', 'Red', 5.0, 70),
        ('Blueberry', 'Blue', 6.0, 50),
    ]
    
    cursor.executemany(
        'INSERT INTO fruit_table (name, color, price, stock) VALUES (?, ?, ?, ?)',
        fruits_data
    )
    
    conn.commit()
    conn.close()
    
    print("✓ Database created!\n")

def get_database_schema():
    conn = sqlite3.connect("fruits_v2.db")
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(fruit_table)")
    columns = cursor.fetchall()

    cursor.execute("SELECT * FROM fruit_table LIMIT 3")
    samples = cursor.fetchall()

    conn.close()

    schema = "Table: fruit_table\n\nColumns:\n"
    for col in columns:
        schema += f"- {col[1]} ({col[2]})\n"
    
    schema += "\nSample rows:\n"
    for row in samples:
        schema += f"- {row}\n"
    
    return schema

# ============================================================
# TOOLS
# ============================================================

@tool
def get_schema() -> str:
    """Get the database schema to understand table structure."""
    schema = get_database_schema()
    print("\n📋 Schema Tool: Providing database schema")
    return schema

@tool
def execute_sql(sql_query: str) -> str:
    """
    Execute a SQL query on the fruit database.
    Input should be a valid SQL SELECT statement.
    
    IMPORTANT: Only SELECT queries are allowed (no INSERT, UPDATE, DELETE).
    """
    print(f"\n💾 SQL Execution Tool: Running query")
    print(f"   Query: {sql_query[:100]}...")
    
    sql_upper = sql_query.upper().strip()
    if not sql_upper.startswith('SELECT'):
        return "Error: Only SELECT queries are allowed for safety."
    
    if any(keyword in sql_upper for keyword in ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER']):
        return "Error: Modification queries are not allowed."
    
    try:
        conn = sqlite3.connect('fruits_v2.db')
        cursor = conn.cursor()
        
        cursor.execute(sql_query)
        results = cursor.fetchall()
        
        column_names = [description[0] for description in cursor.description]
        
        conn.close()
        
        if not results:
            return "Query executed successfully but returned no results."
        
        output = f"Columns: {', '.join(column_names)}\n\n"
        output += f"Results ({len(results)} rows):\n"
        
        for row in results:
            output += f"  {row}\n"
        
        print(f"   ✓ Returned {len(results)} rows")
        return output
    
    except sqlite3.Error as e:
        error_msg = f"SQL Error: {str(e)}"
        print(f"   ❌ {error_msg}")
        return error_msg

@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations."""
    print(f"\n🧮 Calculator: {expression}")
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        print(f"   ✓ Result: {result}")
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool
def get_weather(city: str) -> str:
    """Get weather for a city (mock)."""
    print(f"\n🌤️  Weather: {city}")
    weather_db = {
        "new york": "Sunny, 72°F",
        "london": "Cloudy, 15°C",
        "paris": "Partly cloudy, 18°C",
    }
    return weather_db.get(city.lower(), f"Weather in {city}: Sunny, 22°C")

# ============================================================
# LLM SETUP
# ============================================================

def create_huggingface_llm(model_key: str):
    """Create HuggingFace LLM via Inference API (FREE)"""
    
    model_config = MODEL_CONFIGS[model_key]
    
    print(f"\n🤗 Initializing HuggingFace Model:")
    print(f"   Model: {model_config['name']}")
    print(f"   Description: {model_config['description']}")
    print(f"   Max tokens: {model_config['max_tokens']}")
    
    llm = HuggingFaceEndpoint(
        repo_id=model_config['name'],
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
        max_new_tokens=model_config['max_tokens'],
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.1,
    )
    
    chat_llm = ChatHuggingFace(llm=llm)
    
    print("✓ HuggingFace LLM initialized!\n")
    return chat_llm

# ============================================================
# SIMPLER STATEFUL AGENT (MANUAL MEMORY MANAGEMENT)
# ============================================================

def create_sql_agent():
    """Create agent - we'll manage memory manually"""
    print("Initializing SQL agent with MEMORY...")
    
    chatllm = create_huggingface_llm(SELECTED_MODEL)
    
    tools = [
        get_schema,
        execute_sql,
        calculator,
        get_weather
    ]
    
    # Create base agent (no memory wrapper)
    agent = create_agent(
        model=chatllm,
        tools=tools,
        system_prompt="""You are a SQL expert assistant with access to a fruit database.

When answering questions about the database:
1. FIRST call get_schema() to see the database structure
2. THEN write a SQL SELECT query to answer the question
3. Call execute_sql() with your generated query
4. Interpret the results and answer in natural language

IMPORTANT SQL RULES:
- Always use SELECT queries only
- Use proper SQL syntax (SQLite)
- Handle NULL values appropriately
- Use LIMIT when appropriate

Example workflow:
User: "What fruits are red?"
1. Call get_schema() to see columns
2. Generate: SELECT name, color, price, stock FROM fruit_table WHERE color = 'Red'
3. Call execute_sql() with that query
4. Format results naturally

You can also use calculator and get_weather tools when needed.

IMPORTANT: When asked to calculate, use the calculator tool for accuracy.

MEMORY: You have access to conversation history. When users refer to "it", "that", or "the previous", 
check the conversation history for context."""
    )
    
    print("✓ SQL Agent with MEMORY ready!\n")
    return agent

def run_query(agent, user_query: str):
    """Run a user query with MANUAL memory management"""
    print("="*70)
    print(f"USER QUERY: {user_query}")
    print("="*70)
    
    try:
        # Create user message
        user_message = HumanMessage(content=user_query)
        
        # Manually add user message to history
        conversation_history.add_message(user_message)
        
        # Prepare messages: history + new message
        all_messages = conversation_history.messages.copy()
        
        print(f"\n[DEBUG] Sending {len(all_messages)} messages to agent")
        
        # Invoke agent with full history
        response = agent.invoke({"messages": all_messages})
        
        print("\n" + "="*40)
        print(f"\n📜 FULL RESPONSE TYPE: {type(response)}")
        print(f"📜 RESPONSE KEYS: {response.keys() if isinstance(response, dict) else 'N/A'}")
        print("\n" + "="*40)

        # Extract new messages from response
        if "messages" in response:
            response_messages = response["messages"]
            
            # Find new messages (ones not already in history)
            new_messages = response_messages[len(all_messages):]
            
            # Add new agent messages to history
            for msg in new_messages:
                if not isinstance(msg, HumanMessage):  # Don't re-add user messages
                    conversation_history.add_message(msg)
            
            final_message = response_messages[-1]
        else:
            final_message = response
            conversation_history.add_message(AIMessage(content=str(response)))
        
        print("\n" + "="*70)
        print("AGENT RESPONSE:")
        print("="*70)
        print(final_message.content if hasattr(final_message, 'content') else final_message)
        print("="*70 + "\n")
        
        # Show conversation history
        print("\n" + "💭"*35)
        print("CONVERSATION HISTORY")
        print("💭"*35)
        print(f"Total messages in memory: {len(conversation_history.messages)}")
        
        for i, msg in enumerate(conversation_history.messages[-10:], 1):  # Last 10 messages
            msg_type = "USER" if isinstance(msg, HumanMessage) else "AGENT"
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            print(f"{i}. [{msg_type}]: {content}")
        print("="*70 + "\n")
        
        return final_message.content if hasattr(final_message, 'content') else str(final_message)
    
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        return None

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STATEFUL SQL AGENT WITH SHORT-TERM MEMORY")
    print(f"Using: {MODEL_CONFIGS[SELECTED_MODEL]['name']}")
    print("="*70 + "\n")
    
    create_database()
    
    print("Database schema:")
    print(get_database_schema())
    
    agent = create_sql_agent()
    
    print("\n" + "="*70)
    print("INTERACTIVE MODE WITH MEMORY")
    print("="*70)
    print("The agent now remembers your conversation!")
    print("Try follow-up questions like:")
    print('  - "What about the price?" (after asking about a fruit)')
    print('  - "Calculate the square of that" (after seeing a number)')
    print('  - "What did I ask about earlier?"')
    print("\nCommands:")
    print("  - Type 'clear' to clear conversation memory")
    print("  - Press Enter (empty) to quit")
    print("="*70 + "\n")
    
    while True:
        user_query = input("🔍 Your query: ")
        
        if user_query.strip() == "":
            print("\n👋 Exiting...")
            break
        
        if user_query.strip().lower() == "clear":
            conversation_history.clear()
            print("\n🧹 Conversation memory cleared!\n")
            continue
        
        print(f"\n{'-'*70}")
        run_query(agent, user_query)
