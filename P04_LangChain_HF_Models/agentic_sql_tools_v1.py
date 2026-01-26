import sqlite3
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.agents import create_agent
from langchain_core.tools import tool
import os
from dotenv import load_dotenv

# ============================================================
# CONFIGURATION: Choose Your Model
# ============================================================

# Get HuggingFace API token (FREE - sign up at huggingface.co)
# Create token at: https://huggingface.co/settings/tokens
# HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "your-token-here")
# loads .env into environment variables
load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Model selection (pick one):
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

# SELECT YOUR MODEL HERE
SELECTED_MODEL = "qwen"  # Change to: "mixtral", "llama", "command-r", or "phi"

# ============================================================
# PART 1: DATABASE SETUP
# ============================================================

def create_database():
    """Create SQLite database with fruit table"""
    print("Creating database...")
    
    conn = sqlite3.connect('fruits_v1.db')
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
    conn = sqlite3.connect('fruits_v1.db')
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA table_info(fruit_table)")
    columns = cursor.fetchall()
    
    cursor.execute("SELECT * FROM fruit_table LIMIT 3")
    samples = cursor.fetchall()
    
    conn.close()
    
    schema_info = """
Database: fruits_v1.db
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
# PART 2: LLM-POWERED SQL TOOLS
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
        conn = sqlite3.connect('fruits_v1.db')
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
# PART 3: HUGGINGFACE LLM SETUP
# ============================================================

def create_huggingface_llm(model_key: str):
    """Create HuggingFace LLM via Inference API (FREE)"""
    
    model_config = MODEL_CONFIGS[model_key]
    
    print(f"\n🤗 Initializing HuggingFace Model:")
    print(f"   Model: {model_config['name']}")
    print(f"   Description: {model_config['description']}")
    print(f"   Max tokens: {model_config['max_tokens']}")
    
    # Check API token
    if HUGGINGFACE_API_TOKEN == "your-token-here":
        print("\n⚠️  WARNING: Please set HUGGINGFACE_API_TOKEN!")
        print("   1. Go to https://huggingface.co/settings/tokens")
        print("   2. Create a token (FREE)")
        print("   3. Set it in code or environment variable:")
        print('      export HUGGINGFACE_API_TOKEN="your-token"')
        print("\n   Using demo mode with limited functionality...")
    
    # Create HuggingFace endpoint
    llm = HuggingFaceEndpoint(
        repo_id=model_config['name'],
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
        max_new_tokens=model_config['max_tokens'],
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.1,
    )
    
    # Wrap in ChatHuggingFace for chat interface
    chat_llm = ChatHuggingFace(llm=llm)
    
    print("✓ HuggingFace LLM initialized!\n")
    return chat_llm

# ============================================================
# PART 4: SQL AGENT WITH HUGGINGFACE
# ============================================================

def create_sql_agent():
    """Create agent that generates SQL queries using HuggingFace LLM"""
    print("Initializing SQL agent with HuggingFace...")
    
    # Create HuggingFace LLM
    llm = create_huggingface_llm(SELECTED_MODEL)
    
    tools = [
        get_schema,
        execute_sql,
        calculator,
        get_weather
    ]
    
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

You can also use calculator and get_weather tools when needed.

IMPORTANT: When asked to calculate, use the calculator tool for accuracy."""
    )
    
    print("✓ SQL Agent ready!\n")
    return agent

def run_query(agent, user_query: str):
    """Run a user query through the SQL agent"""
    print("="*70)
    print(f"USER QUERY: {user_query}")
    print("="*70)
    
    try:
        response = agent.invoke({
            "messages": [{"role": "user", "content": user_query}]
        })
        
        final_message = response["messages"][-1]
        
        print("\n" + "="*70)
        print("AGENT RESPONSE:")
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
    print("SQL AGENT WITH HUGGINGFACE LLM")
    print(f"Using: {MODEL_CONFIGS[SELECTED_MODEL]['name']}")
    print("="*70 + "\n")
    
    # Create database
    create_database()
    
    # Verify
    print("Database schema:")
    print(get_database_schema())
    
    # Create agent
    agent = create_sql_agent()
    
    # Interactive mode
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("Enter your queries. Press Enter with empty input to quit.")
    # user_query = "What is the stock value of fruit name Mango, calculate the square of that value using calculator"
    print("="*70 + "\n")
    
    while True:
        user_query = input("🔍 Your query: ")
        if user_query.strip() == "":
            print("\n👋 Exiting...")
            break
        
        print(f"\n{'-'*70}")
        run_query(agent, user_query)
