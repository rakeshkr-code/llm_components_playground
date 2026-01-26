import sqlite3
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.tools import tool

# ============================================================
# PART 1: DATABASE CREATION & VERIFICATION
# ============================================================

def create_database():
    """Create SQLite database with fruit table"""
    print("Creating database...")
    
    # Connect to SQLite (creates file if doesn't exist)
    conn = sqlite3.connect('fruits.db')
    cursor = conn.cursor()
    
    # Drop table if exists (fresh start)
    cursor.execute('DROP TABLE IF EXISTS fruit_table')
    
    # Create table
    cursor.execute('''
        CREATE TABLE fruit_table (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            color TEXT NOT NULL,
            price REAL,
            stock INTEGER
        )
    ''')
    
    # Insert sample data
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
    
    print("✓ Database created successfully!\n")


def verify_database():
    """Verify database contents"""
    print("Verifying database...")
    print("="*60)
    
    conn = sqlite3.connect('fruits.db')
    cursor = conn.cursor()
    
    # Get all data
    cursor.execute('SELECT * FROM fruit_table')
    rows = cursor.fetchall()
    
    print(f"Total fruits in database: {len(rows)}\n")
    print(f"{'ID':<5} {'Name':<15} {'Color':<10} {'Price':<8} {'Stock':<8}")
    print("-"*60)
    
    for row in rows:
        print(f"{row[0]:<5} {row[1]:<15} {row[2]:<10} ${row[3]:<7.2f} {row[4]:<8}")
    
    print("="*60 + "\n")
    
    conn.close()


# ============================================================
# PART 2: TOOL DEFINITIONS
# ============================================================

@tool
def query_database(question: str) -> str:
    """
    Query the fruit database. Use natural language to ask about fruits.
    Examples: 
    - "What fruits are red?"
    - "Show me all yellow fruits"
    - "What's the price of apples?"
    - "Which fruit has the highest stock?"
    """
    print(f"\n🔍 Database Tool: Querying for '{question}'")
    
    try:
        conn = sqlite3.connect('fruits.db')
        cursor = conn.cursor()
        
        # Simple keyword-based SQL generation
        question_lower = question.lower()
        
        # Detect what kind of query
        if 'red' in question_lower or 'color' in question_lower:
            if 'red' in question_lower:
                cursor.execute("SELECT name, color, price, stock FROM fruit_table WHERE color = 'Red'")
            elif 'yellow' in question_lower:
                cursor.execute("SELECT name, color, price, stock FROM fruit_table WHERE color = 'Yellow'")
            elif 'green' in question_lower:
                cursor.execute("SELECT name, color, price, stock FROM fruit_table WHERE color = 'Green'")
            else:
                cursor.execute("SELECT name, color, price, stock FROM fruit_table")
        
        elif 'price' in question_lower:
            # Find specific fruit price
            for fruit in ['apple', 'banana', 'kiwi', 'orange', 'grape', 'mango']:
                if fruit in question_lower:
                    cursor.execute(f"SELECT name, price FROM fruit_table WHERE LOWER(name) = '{fruit}'")
                    break
            else:
                cursor.execute("SELECT name, price FROM fruit_table ORDER BY price DESC")
        
        elif 'stock' in question_lower or 'inventory' in question_lower:
            if 'highest' in question_lower or 'most' in question_lower:
                cursor.execute("SELECT name, stock FROM fruit_table ORDER BY stock DESC LIMIT 1")
            else:
                cursor.execute("SELECT name, stock FROM fruit_table ORDER BY stock DESC")
        
        elif 'all' in question_lower or 'list' in question_lower:
            cursor.execute("SELECT name, color, price, stock FROM fruit_table")
        
        else:
            # Default: try to find fruit by name
            cursor.execute("SELECT name, color, price, stock FROM fruit_table")
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return "No fruits found matching your query."
        
        # Format results
        output = []
        for row in results:
            if len(row) == 2:
                output.append(f"{row[0]}: {row[1]}")
            else:
                output.append(f"{row[0]} ({row[1]}) - Price: ${row[2]}, Stock: {row[3]}")
        
        result_text = "\n".join(output)
        print(f"   ✓ Found {len(results)} result(s)")
        return result_text
    
    except Exception as e:
        return f"Database error: {str(e)}"


@tool
def calculator(expression: str) -> str:
    """
    Perform mathematical calculations.
    Examples: "10 * 5", "2.5 + 3.0", "(100 - 20) / 2"
    """
    print(f"\n🧮 Calculator Tool: Calculating '{expression}'")
    
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        print(f"   ✓ Result: {result}")
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def get_weather(city: str) -> str:
    """
    Get weather information for a city (mock data).
    Examples: "New York", "London", "Tokyo"
    """
    print(f"\n🌤️  Weather Tool: Getting weather for '{city}'")
    
    # Mock weather data
    weather_db = {
        "new york": "Sunny, 72°F",
        "london": "Cloudy, 15°C",
        "tokyo": "Rainy, 20°C",
        "paris": "Partly cloudy, 18°C",
        "sydney": "Clear, 25°C",
    }
    
    city_lower = city.lower()
    if city_lower in weather_db:
        weather = weather_db[city_lower]
        print(f"   ✓ Weather: {weather}")
        return f"Weather in {city}: {weather}"
    else:
        return f"Weather in {city}: Sunny, 22°C (default)"


@tool
def fruit_recommendation(criteria: str) -> str:
    """
    Recommend fruits based on criteria like budget, color preference, or health goals.
    Examples: "cheap fruits", "colorful fruits", "high stock items"
    """
    print(f"\n🍎 Recommendation Tool: Finding fruits matching '{criteria}'")
    
    conn = sqlite3.connect('fruits.db')
    cursor = conn.cursor()
    
    criteria_lower = criteria.lower()
    
    if 'cheap' in criteria_lower or 'budget' in criteria_lower:
        cursor.execute("SELECT name, price FROM fruit_table WHERE price < 3.0 ORDER BY price")
    elif 'expensive' in criteria_lower or 'premium' in criteria_lower:
        cursor.execute("SELECT name, price FROM fruit_table WHERE price > 4.0 ORDER BY price DESC")
    elif 'colorful' in criteria_lower:
        cursor.execute("SELECT name, color FROM fruit_table WHERE color IN ('Red', 'Purple', 'Orange', 'Yellow')")
    elif 'stock' in criteria_lower or 'available' in criteria_lower:
        cursor.execute("SELECT name, stock FROM fruit_table WHERE stock > 100 ORDER BY stock DESC")
    else:
        cursor.execute("SELECT name, color, price FROM fruit_table LIMIT 3")
    
    results = cursor.fetchall()
    conn.close()
    
    if not results:
        return "No recommendations found."
    
    recommendations = [f"{row[0]} - {row[1]}" for row in results]
    print(f"   ✓ Found {len(recommendations)} recommendations")
    return "Recommended fruits: " + ", ".join(recommendations)


# ============================================================
# PART 3: AGENT SETUP
# ============================================================

def create_fruit_agent():
    """Create agent with all tools"""
    print("Initializing agent with tools...")
    
    # Initialize LLM
    llm = ChatOllama(model="llama3.2:1b", temperature=0)
    
    # All tools
    tools = [
        query_database,
        calculator,
        get_weather,
        fruit_recommendation
    ]
    
    # Create agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""You are a helpful fruit store assistant. You have access to:
1. query_database: Search fruit inventory (name, color, price, stock)
2. calculator: Perform calculations for totals, discounts, etc.
3. get_weather: Get weather info (for delivery planning)
4. fruit_recommendation: Suggest fruits based on criteria

Always use tools to get accurate information. Combine multiple tools when needed."""
    )
    
    print("✓ Agent ready!\n")
    return agent


def run_query(agent, user_query: str):
    """Run a user query through the agent"""
    print("="*60)
    print(f"USER QUERY: {user_query}")
    print("="*60)
    
    try:
        response = agent.invoke({
            "messages": [{"role": "user", "content": user_query}]
        })
        
        # Extract final answer
        final_message = response["messages"][-1]
        
        print("\n" + "="*60)
        print("AGENT RESPONSE:")
        print("="*60)
        print(final_message.content)
        print("="*60 + "\n")
        
        return final_message.content
    
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        return None


# ============================================================
# PART 4: MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("FRUIT STORE ASSISTANT WITH DATABASE & TOOLS")
    print("="*60 + "\n")
    
    # Step 1: Create and verify database
    create_database()
    verify_database()
    
    # Step 2: Create agent
    agent = create_fruit_agent()
    
    # Step 3: Test queries
    test_queries = [
        # Simple database query
        "What fruits are red?",
        
        # Database + Calculator
        "If I buy 5 apples at their current price, what's the total cost?",
        
        # Database + Weather
        "What yellow fruits do we have? Also, what's the weather in London?",
        
        # Multiple tools
        "Recommend cheap fruits, calculate the total if I buy 3 of each, and check if delivery weather in Paris is good",
        
        # Complex query
        "Find all fruits under $3, calculate 20% discount on the most expensive one, and recommend colorful options"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'#'*60}")
        print(f"TEST QUERY {i}/{len(test_queries)}")
        print(f"{'#'*60}\n")
        
        run_query(agent, query)
        
        # Pause between queries
        if i < len(test_queries):
            input("Press Enter for next query...")
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETE!")
    print("="*60)
