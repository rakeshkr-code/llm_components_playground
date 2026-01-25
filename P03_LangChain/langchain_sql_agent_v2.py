import sqlite3
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.tools import tool

# ============================================================
# PART 1: DATABASE SETUP (Same as before)
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
    """Get database schema for LLM"""
    conn = sqlite3.connect('fruits_v2.db')
    cursor = conn.cursor()
    
    # Get table schema
    cursor.execute("PRAGMA table_info(fruit_table)")
    columns = cursor.fetchall()
    
    # Get sample data
    cursor.execute("SELECT * FROM fruit_table LIMIT 3")
    samples = cursor.fetchall()
    
    conn.close()
    
    # Format schema for LLM
    schema_info = """
Database: fruits_v2.db
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
    
    # Safety check
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
        
        # Get column names
        column_names = [description[0] for description in cursor.description]
        
        conn.close()
        
        if not results:
            return "Query executed successfully but returned no results."
        
        # Format results
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
# PART 3: LLM SQL AGENT
# ============================================================

def create_sql_agent():
    """Create agent that generates SQL queries using LLM"""
    print("Initializing SQL agent with LLM...")
    
    llm = ChatOllama(model="llama3.2:1b", temperature=0)
    
    tools = [
        get_schema,      # LLM calls this to understand database
        execute_sql,     # LLM generates SQL and executes via this
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

You can also use calculator and get_weather tools when needed."""
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
        return None


# ============================================================
# PART 4: MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("SQL AGENT WITH LLM-GENERATED QUERIES")
    print("="*70 + "\n")
    
    # Create database
    create_database()
    
    # Verify
    print("Database schema:")
    print(get_database_schema())
    
    # Create agent
    agent = create_sql_agent()
    
    # Test queries
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
            input("Press Enter for next test...")
    
    print("\n✅ All tests complete!")
