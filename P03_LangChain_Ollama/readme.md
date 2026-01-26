Langchain SQL agent py has n versions:

- V1: Here agent has multiple tools but here LLM is not generating the query, few structured query is pre planned and provided as tools (functions)
- V2: This is the main implementation where the LLM is actually generating the query and tool is getting the result from db
- V3: Added some callbacks to the V2 for printing or logging the internal conversation inside agent, ie, llm vs tools vs agent

* V2 id the main till now

