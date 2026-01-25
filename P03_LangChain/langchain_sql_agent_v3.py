import sqlite3
from langchain_core.callbacks import BaseCallbackHandler
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.tools import tool
from typing import Any, Dict, List
import json
from pprint import pprint, pformat
from datetime import datetime
import sys

# ============================================================
# LOGGING SETUP
# ============================================================

class DualOutput:
    """Write to both console and file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate write
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

# Create log file with timestamp
log_filename = f"agent_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
dual_output = DualOutput(log_filename)
sys.stdout = dual_output

print(f"📝 Logging to: {log_filename}\n")

# ============================================================
# IMPROVED VERBOSE CALLBACK - LOGS EVERYTHING!
# ============================================================

class VerboseAgentCallback(BaseCallbackHandler):
    """Custom callback to see and log every step"""
    
    def __init__(self):
        self.step_number = 0
        self.tool_call_number = 0
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts processing"""
        self.step_number += 1
        print("\n" + "🔵"*35)
        print(f"STEP {self.step_number}: LLM THINKING")
        print("🔵"*35)
        
        print("\n📨 FULL PROMPT SENT TO MODEL:")
        print("─"*70)
        print(prompts[0])
        print("─"*70)
        print(f"Prompt length: {len(prompts[0])} characters")
        
        print("\n📋 SERIALIZED LLM INFO:")
        print("─"*70)
        pprint(serialized, width=100)
        
        print("\n📦 KWARGS:")
        print("─"*70)
        pprint(kwargs, width=100)
        print("─"*70)
    
    def on_llm_end(self, response, **kwargs) -> None:
        """Called when LLM finishes - LOG EVERYTHING!"""
        print("\n✅ MODEL RESPONSE (COMPLETE OBJECT):")
        print("="*70)
        
        # Print ENTIRE response object using pprint
        print("\n🔍 FULL RESPONSE OBJECT:")
        print("─"*70)
        
        # Convert response to dict for better inspection
        try:
            response_dict = {
                "llm_output": response.llm_output,
                "run": str(response.run) if hasattr(response, 'run') else None,
            }
            print("Response metadata:")
            pprint(response_dict, width=100)
        except:
            pass
        
        # Print generations
        print("\n📊 GENERATIONS:")
        print("─"*70)
        for i, generation_list in enumerate(response.generations):
            print(f"\nGeneration List {i}:")
            for j, gen in enumerate(generation_list):
                print(f"\n  Generation {j}:")
                print(f"    Type: {type(gen).__name__}")
                
                # Print all attributes
                print(f"\n    All attributes:")
                pprint(vars(gen), width=100)
                
                # Check for text
                if hasattr(gen, 'text'):
                    print(f"\n    Text: {gen.text}")
                
                # Check for message (THIS IS WHERE TOOL CALLS ARE!)
                if hasattr(gen, 'message'):
                    print(f"\n    Message object:")
                    msg = gen.message
                    print(f"      Type: {type(msg).__name__}")
                    print(f"      Content: {msg.content}")
                    
                    # TOOL CALLS!
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        print(f"\n      🛠️  TOOL CALLS (AGENT DECISIONS):")
                        for k, tc in enumerate(msg.tool_calls, 1):
                            print(f"\n        Tool Call {k}:")
                            pprint(tc, width=100, indent=10)
                    
                    # Additional kwargs
                    if hasattr(msg, 'additional_kwargs'):
                        print(f"\n      Additional kwargs:")
                        pprint(msg.additional_kwargs, width=100, indent=10)
                    
                    # All message attributes
                    print(f"\n      All message attributes:")
                    pprint(vars(msg), width=100, indent=10)
        
        print("\n" + "="*70)
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when a tool is about to be executed"""
        self.tool_call_number += 1
        tool_name = serialized.get('name', 'Unknown')
        
        print("\n" + "🟢"*35)
        print(f"TOOL EXECUTION #{self.tool_call_number}: {tool_name}")
        print("🟢"*35)
        
        print(f"\n📥 Tool Input (raw):")
        print("─"*70)
        print(f"Type: {type(input_str)}")
        print(f"Value: {input_str}")
        
        print(f"\n📋 Serialized Tool Info:")
        print("─"*70)
        pprint(serialized, width=100)
        
        print(f"\n📦 Kwargs:")
        print("─"*70)
        pprint(kwargs, width=100)
        print("─"*70)
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool finishes execution"""
        print(f"\n📤 Tool Output:")
        print("─"*70)
        if len(output) > 500:
            print(output[:500])
            print(f"\n... (truncated, total: {len(output)} chars)")
        else:
            print(output)
        
        print(f"\n📦 Kwargs:")
        pprint(kwargs, width=100)
        print("─"*70)
    
    def on_tool_error(self, error, **kwargs) -> None:
        """Called when a tool encounters an error"""
        print(f"\n❌ TOOL ERROR:")
        print("─"*70)
        print(f"Error: {error}")
        pprint(kwargs, width=100)
        print("─"*70)
    
    def on_agent_action(self, action, **kwargs):
        """Called when agent decides on an action"""
        print("\n" + "🧠"*35)
        print("AGENT DECISION (RAW)")
        print("🧠"*35)
        
        print("\n📊 Action Object:")
        print("─"*70)
        pprint(vars(action), width=100)
        
        print(f"\n📦 Kwargs:")
        pprint(kwargs, width=100)
        print("─"*70)
    
    def on_agent_finish(self, finish, **kwargs):
        """Called when agent finishes"""
        print("\n" + "🏁"*35)
        print("AGENT FINISHED")
        print("🏁"*35)
        
        print("\n📊 Finish Object:")
        print("─"*70)
        pprint(vars(finish), width=100)
        
        print(f"\n📦 Kwargs:")
        pprint(kwargs, width=100)
        print("─"*70)
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        """Called when a chain starts"""
        print("\n" + "⛓️ "*35)
        print("CHAIN STARTED")
        print("⛓️ "*35)
        
        print(f"\n📋 Serialized:")
        pprint(serialized, width=100)
        
        print(f"\n📥 Inputs:")
        pprint(inputs, width=100)
        
        print(f"\n📦 Kwargs:")
        pprint(kwargs, width=100)
        print("─"*70)
    
    def on_chain_end(self, outputs, **kwargs):
        """Called when a chain ends"""
        print("\n" + "⛓️ "*35)
        print("CHAIN ENDED")
        print("⛓️ "*35)
        
        print(f"\n📤 Outputs:")
        pprint(outputs, width=100)
        
        print(f"\n📦 Kwargs:")
        pprint(kwargs, width=100)
        print("─"*70)

# ============================================================
# DATABASE SETUP
# ============================================================

def create_database():
    """Create SQLite database with fruit table"""
    print("Creating database...")
    
    conn = sqlite3.connect('fruits_v3.db')
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
    """Get database schema for LLM"""
    conn = sqlite3.connect('fruits_v3.db')
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA table_info(fruit_table)")
    columns = cursor.fetchall()
    
    cursor.execute("SELECT * FROM fruit_table LIMIT 3")
    samples = cursor.fetchall()
    
    conn.close()
    
    schema_info = """
Database: fruits_v3.db
Table: fruit_table

Columns:
- id: INTEGER (Primary Key)
- name: TEXT (Fruit name)
- color: TEXT (Fruit color)
- price: REAL (Price in dollars)
- stock: INTEGER (Available quantity)

Sample data:
"""
    for sample in samples:
        schema_info += f"- {sample}\n"
    
    return schema_info

# ============================================================
# TOOLS
# ============================================================

@tool
def get_schema() -> str:
    """Get the database schema to understand table structure."""
    schema = get_database_schema()
    print("\n📋 [Tool Executing] get_schema()")
    return schema

@tool
def execute_sql(sql_query: str) -> str:
    """
    Execute a SQL query on the fruit database.
    Input should be a valid SQL SELECT statement.
    
    IMPORTANT: Only SELECT queries are allowed (no INSERT, UPDATE, DELETE).
    """
    print(f"\n💾 [Tool Executing] execute_sql()")
    print(f"   SQL: {sql_query}")
    
    sql_upper = sql_query.upper().strip()
    if not sql_upper.startswith('SELECT'):
        return "Error: Only SELECT queries are allowed for safety."
    
    if any(keyword in sql_upper for keyword in ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER']):
        return "Error: Modification queries are not allowed."
    
    try:
        conn = sqlite3.connect('fruits_v3.db')
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
    print(f"\n🧮 [Tool Executing] calculator({expression})")
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        print(f"   ✓ Result: {result}")
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool
def get_weather(city: str) -> str:
    """Get weather for a city (mock)."""
    print(f"\n🌤️  [Tool Executing] get_weather({city})")
    weather_db = {
        "new york": "Sunny, 72°F",
        "london": "Cloudy, 15°C",
        "paris": "Partly cloudy, 18°C",
    }
    return weather_db.get(city.lower(), f"Weather in {city}: Sunny, 22°C")

# ============================================================
# AGENT
# ============================================================

def create_sql_agent():
    """Create agent that generates SQL queries using LLM"""
    print("Initializing SQL agent with LLM...")
    
    verbose_callback = VerboseAgentCallback()
    
    llm = ChatOllama(
        model="llama3.2:1b",
        temperature=0,
        callbacks=[verbose_callback]
    )
    
    tools = [get_schema, execute_sql, calculator, get_weather]
    
    agent = create_agent(
        model=llm,
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
- Join tables if needed (though we only have one table)

Example workflow:
User: "What fruits are red?"
1. Call get_schema() to see columns
2. Generate: SELECT name, color, price, stock FROM fruit_table WHERE color = 'Red'
3. Call execute_sql() with that query
4. Format results naturally

You can also use calculator and get_weather tools when needed."""
    )
    
    print("✓ SQL Agent ready!\n")
    return agent

def run_query(agent, user_query: str):
    """Run a user query through the SQL agent"""
    print("\n" + "="*70)
    print(f"USER QUERY: {user_query}")
    print("="*70)
    
    try:
        response = agent.invoke({
            "messages": [{"role": "user", "content": user_query}]
        })
        
        # Print COMPLETE response object
        print("\n" + "📦"*35)
        print("COMPLETE RESPONSE OBJECT (PPRINT)")
        print("📦"*35)
        print("\nResponse type:", type(response))
        print("\nResponse keys:", list(response.keys()) if isinstance(response, dict) else "N/A")
        print("\n" + "─"*70)
        pprint(response, width=120)
        print("─"*70)
        
        # Print message history with ALL details
        print("\n" + "📜"*35)
        print("COMPLETE MESSAGE HISTORY (DETAILED)")
        print("📜"*35)
        
        for i, msg in enumerate(response["messages"]):
            print(f"\n{'='*70}")
            print(f"Message {i+1}: {type(msg).__name__}")
            print('='*70)
            
            # Print ALL attributes using pprint
            print("\n🔍 All Message Attributes:")
            print("─"*70)
            pprint(vars(msg), width=120)
            print("─"*70)
            
            # Specifically highlight content
            if msg.content:
                print(f"\n📝 Content:\n{msg.content}")
            
            # Specifically highlight tool calls
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print("\n🛠️  Tool Calls:")
                print("─"*70)
                pprint(msg.tool_calls, width=120)
                print("─"*70)
        
        # Extract final answer
        final_message = response["messages"][-1]
        
        print("\n" + "🎯"*35)
        print("FINAL ANSWER")
        print("🎯"*35)
        print(final_message.content)
        print("="*70 + "\n")
        
        return final_message.content
    
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        return None

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    try:
        print("\n" + "="*70)
        print("SQL AGENT WITH COMPLETE VERBOSE LOGGING")
        print("ALL OUTPUT LOGGED TO FILE")
        print("="*70 + "\n")
        
        # Create database
        create_database()
        
        # Verify
        print("Database schema:")
        print(get_database_schema())
        
        # Create agent
        agent = create_sql_agent()
        
        # Test queries (reduced for initial testing)
        test_queries = [
            # Simple query
            "What fruits are red?",
            
            # Range query
            "Show me fruits between $2 and $4",
            
            # Aggregate query
            "What's the average price of all fruits?",
            
            # Complex query
            "Find fruits with stock above 80 and price below $3",
            
            # Multi-tool query
            "List all yellow fruits, calculate total value (price * stock) for bananas, and check London weather",
            
            # Advanced query
            "Which fruit has the best price-to-stock ratio?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'#'*70}")
            print(f"TEST {i}/{len(test_queries)}")
            print(f"{'#'*70}\n")
            
            run_query(agent, query)
            
            if i < len(test_queries):
                # Don't wait for input, just continue
                print("\n" + "─"*70)
                print("Moving to next test...")
                print("─"*70)
        
        print("\n✅ All tests complete!")
        print(f"\n📝 Full log saved to: {log_filename}")
        
    finally:
        # Close log file
        sys.stdout = dual_output.terminal
        dual_output.close()
        print(f"\n✅ Log file closed: {log_filename}")
