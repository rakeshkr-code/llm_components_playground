import sqlite3
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import SystemMessage
import os
from dotenv import load_dotenv
import json

# ============================================================
# CONFIGURATION: Choose Your Model
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
# REQUIREMENT 1: PROMPT LOGGING CALLBACK
# ============================================================

class PromptLoggingCallback(BaseCallbackHandler):
    """Callback handler to log actual prompts sent to LLM"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.call_count = 0
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called when LLM starts - logs the actual prompt"""
        self.call_count += 1
        
        if self.verbose:
            print("\n" + "="*80)
            print(f"🔍 LLM INVOCATION #{self.call_count}")
            print("="*80)
            
            # Log prompts (for completion-style models)
            if prompts:
                for i, prompt in enumerate(prompts):
                    print(f"\n📝 PROMPT #{i+1}:")
                    print("-"*80)
                    print(prompt)
                    print("-"*80)
    
    def on_chat_model_start(self, serialized, messages, **kwargs):
        """Called when chat model starts - logs messages"""
        self.call_count += 1
        
        if self.verbose:
            print("\n" + "="*80)
            print(f"🔍 CHAT MODEL INVOCATION #{self.call_count}")
            print("="*80)
            
            # Log each message in the conversation
            for msg_list in messages:
                for msg in msg_list:
                    role = msg.__class__.__name__
                    content = msg.content
                    
                    print(f"\n📨 {role.upper()}:")
                    print("-"*80)
                    print(content)
                    
                    # Log tool calls if present
                    if hasattr(msg, 'additional_kwargs') and 'tool_calls' in msg.additional_kwargs:
                        print("\n🛠️  TOOL CALLS:")
                        print(json.dumps(msg.additional_kwargs['tool_calls'], indent=2))
                    
                    print("-"*80)

# ============================================================
# PART 1: DATABASE SETUP
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
    """Get database schema with detailed information"""
    conn = sqlite3.connect("fruits_v2.db")
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(fruit_table)")
    columns = cursor.fetchall()

    cursor.execute("SELECT * FROM fruit_table")
    all_rows = cursor.fetchall()

    conn.close()

    schema = "DATABASE SCHEMA (CACHED)\n"
    schema += "="*60 + "\n"
    schema += "Table: fruit_table\n\n"
    schema += "Columns:\n"
    for col in columns:
        schema += f"  - {col[1]} ({col[2]})"
        if col[5] == 1:
            schema += " [PRIMARY KEY]"
        schema += "\n"
    
    schema += f"\nTotal rows: {len(all_rows)}\n"
    schema += "\nAll data:\n"
    for row in all_rows:
        schema += f"  {row}\n"
    
    schema += "="*60 + "\n"
    
    return schema

# ============================================================
# PART 2: LLM-POWERED SQL TOOLS (MODIFIED - NO get_schema tool)
# ============================================================

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
    """
    Perform mathematical calculations.
    Use Python syntax: +, -, *, /, **, % for arithmetic operations.
    """
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
# PART 3: HUGGINGFACE LLM SETUP
# ============================================================

def create_huggingface_llm(model_name: str):
    """Create HuggingFace LLM via Inference API (FREE)"""
    
    model_config = MODEL_CONFIGS[model_name]
    
    print(f"\n🤗 Initializing HuggingFace Model:")
    print(f"   Model: {model_config['name']}")
    print(f"   Description: {model_config['description']}")
    print(f"   Max tokens: {model_config['max_tokens']}")
    
    if HUGGINGFACE_API_TOKEN == "your-token-here" or not HUGGINGFACE_API_TOKEN:
        print("\n⚠️  WARNING: Please set HUGGINGFACE_API_TOKEN!")
        print("   1. Go to https://huggingface.co/settings/tokens")
        print("   2. Create a token (FREE)")
        print("   3. Set it in .env file or environment variable")
    
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
# REQUIREMENT 2: ENHANCED SYSTEM PROMPT WITH PARALLEL TOOL CALLING
# ============================================================

def create_enhanced_system_prompt(db_schema: str) -> str:
    """
    Create enhanced system prompt with:
    1. Cached database schema
    2. Parallel tool calling instructions
    """
    
    prompt = f"""You are an expert SQL and data analysis assistant with access to a fruit database and calculation tools.

{db_schema}

CRITICAL INSTRUCTION - PARALLEL TOOL EXECUTION:
═══════════════════════════════════════════════════════════════

⚡ WHEN MULTIPLE OPERATIONS ARE INDEPENDENT, CALL ALL TOOLS IN PARALLEL (SAME TURN)

Examples of INDEPENDENT operations (call tools together):
✓ "What is the color of kiwi and calculate 20 * 100?"
  → Call execute_sql() AND calculator() in the SAME response
  
✓ "Get weather in Paris and find all red fruits"
  → Call get_weather() AND execute_sql() in the SAME response

✓ "Calculate 5 * 5 and show me all fruits with stock > 100"
  → Call calculator() AND execute_sql() in the SAME response

Examples of DEPENDENT operations (call tools sequentially):
✓ "What is the square of stock value of Mango?"
  → FIRST call execute_sql() to get stock
  → THEN call calculator() with the result

✓ "Calculate total price if I buy all apples"
  → FIRST call execute_sql() to get price and stock
  → THEN call calculator() with those values

RULE: If operations DON'T depend on each other's outputs → PARALLEL (same turn)
      If one operation NEEDS the output of another → SEQUENTIAL (separate turns)

SQL QUERY GUIDELINES:
═════════════════════
- You already have the complete database schema above (CACHED)
- Write SQL SELECT queries directly - no need to call get_schema()
- Use proper SQLite syntax
- Handle case-insensitive searches: WHERE LOWER(name) = LOWER('Kiwi')
- Use appropriate WHERE, ORDER BY, LIMIT clauses

RESPONSE FORMAT:
════════════════
1. Analyze if operations are independent or dependent
2. If independent: Call all required tools in parallel
3. If dependent: Call tools sequentially, waiting for results
4. Provide natural language summary of all results

AVAILABLE TOOLS:
- execute_sql: Run SELECT queries on the fruit database
- calculator: Perform mathematical calculations
- get_weather: Get weather information for cities

Remember: MAXIMIZE PARALLELISM for independent operations!"""
    
    return prompt

# ============================================================
# PART 4: SQL AGENT WITH ENHANCED PROMPT
# ============================================================

def create_sql_agent():
    """Create agent with enhanced system prompt and logging"""
    print("Initializing SQL agent with enhanced prompt and logging...\n")
    
    # Create HuggingFace LLM
    llm = create_huggingface_llm(SELECTED_MODEL)
    
    # Get database schema ONCE (for prompt caching)
    db_schema = get_database_schema()
    print("📋 Database schema loaded for prompt caching\n")
    
    # Create enhanced system prompt
    system_prompt = create_enhanced_system_prompt(db_schema)
    
    # Tools (NO get_schema tool - schema is in system prompt)
    tools = [
        execute_sql,
        calculator,
        get_weather
    ]
    
    # Create agent with enhanced prompt
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )
    
    print("✓ Enhanced SQL Agent ready with:")
    print("  • Database schema cached in system prompt")
    print("  • Parallel tool calling enabled")
    print("  • Prompt logging instrumented\n")
    
    return agent

def run_query(agent, user_query: str, verbose_logging=True):
    """Run a user query through the SQL agent with prompt logging"""
    print("="*70)
    print(f"USER QUERY: {user_query}")
    print("="*70)
    
    # Create callback for prompt logging
    prompt_logger = PromptLoggingCallback(verbose=verbose_logging)
    
    try:
        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_query}]},
            # config={"callbacks": [prompt_logger]}
        )
        
        print("\n" + "="*70)
        print("📊 TOOL CALL SUMMARY:")
        print("="*70)
        
        # Count tool calls
        tool_calls_per_turn = []
        current_turn_tools = []
        
        for msg in response["messages"]:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                current_turn_tools = [tc['name'] for tc in msg.tool_calls]
                tool_calls_per_turn.append(current_turn_tools)
        
        if tool_calls_per_turn:
            for i, tools in enumerate(tool_calls_per_turn, 1):
                parallel_indicator = "🔥 PARALLEL" if len(tools) > 1 else "⚡ SEQUENTIAL"
                print(f"Turn {i}: {parallel_indicator} → {', '.join(tools)}")
        
        print("="*70 + "\n")

        # -----------------
        print("\n" + "="*40)
        print(f"\n📜 FULL RESPONSE: \n{response}\n")
        print("\n" + "="*40)
        # -----------------
        
        # Extract final response
        final_message = response["messages"][-1]
        
        print("="*70)
        print("🤖 AGENT FINAL RESPONSE:")
        print("="*70)
        print(final_message.content)
        print("="*70 + "\n")
        
        return final_message.content
    
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        return None

# ============================================================
# PART 5: MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 ENHANCED SQL AGENT WITH PARALLEL TOOL CALLING")
    print(f"Using: {MODEL_CONFIGS[SELECTED_MODEL]['name']}")
    print("="*70 + "\n")
    
    # Create database
    create_database()
    
    # Create agent
    agent = create_sql_agent()
    
    # # Test queries demonstrating parallel vs sequential execution
    # print("\n" + "="*70)
    # print("🧪 TEST MODE - PARALLEL vs SEQUENTIAL")
    # print("="*70 + "\n")
    
    # test_queries = [
    #     # Should use PARALLEL tool calling (independent operations)
    #     "What is the color of kiwi and calculate multiplication of 20 and 100?",
        
    #     # Should use SEQUENTIAL tool calling (dependent operations)
    #     "What is the square of the stock value of fruit Mango?",
    # ]
    
    # for query in test_queries:
    #     run_query(agent, query, verbose_logging=True)
    #     print("\n" + "="*70 + "\n")
    
    # Interactive mode
    print("\n" + "="*70)
    print("💬 INTERACTIVE MODE")
    print("="*70)
    print("Enter your queries. Type 'exit' or press Enter to quit.")
    print("Type 'test' to run test queries again.")
    print("="*70 + "\n")
    
    while True:
        user_query = input("🔍 Your query: ").strip()
        
        if user_query == "" or user_query.lower() == "exit":
            print("\n👋 Exiting...")
            break
        
        # if user_query.lower() == "test":
        #     for query in test_queries:
        #         run_query(agent, query, verbose_logging=False)
        #         print("\n")
        #     continue
        
        print(f"\n{'-'*70}")
        run_query(agent, user_query, verbose_logging=False)
