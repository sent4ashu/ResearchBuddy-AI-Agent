""" Tools the agent can call
Design Rules:
1. Each tool is a Python function with a @tool on top.
2. docstring is LLM's guide for when to call this tool and what to pass
3. Type Hints on arguments also for LLM.
4 Neverraise error it will crash the agent loop, instead return error message as string. let agent recover and try something else.
5. keep return value short. Tool's output goes back to LLM's context window and context is expensive.
"""
import wikipedia
from ddgs import DDGS
from langchain_core.tools import tool

# Tool 1: Web search via DuckDuckGo
@tool
def web_search(query: str) -> str:
    """Search the web for current information using DuckDuckGo.

    Use this for questions about recent events, current prices, today's weather,
    news, or anything that requires up-to-date information from the internet.

    Args:
        query: A short, focused search query (1-6 words works best, with a
               proper noun if possible — e.g. "Anthropic Claude release" beats
               "latest AI news").

    Returns:
        A string with the top 3 search result titles and snippets.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results:
                return f"No results found for query: {query}"
            
            # Format results as a concise summary
            formatted = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "No title")
                snippet = r.get("body", "No snippet")
                url = r.get("href", "")
                formatted.append(f"{i}. {title}\n{snippet}\n ({url})")
            
            return "\n\n".join(formatted)
    except Exception as e:
        # Tools never raise errors, return error message instead
        return f"Web Search failed: {type(e).__name__}: {str(e)}"
    

# Tool 2: Wikipedia lookup
@tool
def wikipedia_lookup(topic: str) -> str:
    """ Look up factual information on Wikipedia. Use this for general knowledge questions, historical facts, scientific concepts, etc. 
    prefer this over web search for well-known topics as Wikipedia is more concise and factual.
    Arguments:
        topic (str): the name of person, place, event, or concept to look up on Wikipedia.
    Returns:
        str: The summary of the Wikipedia page.
    """
    try:
        summary = wikipedia.summary(topic, sentences=3, auto_suggest=True)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        # On exception return the options so the agent re-query with a more specific topic
        options = ", ".join(e.options[:5])  # show top 5 options
        return f"Topic '{topic} is ambiguous. Did you mean: {options}?"
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for topic: {topic}"
    except Exception as e:
        return f"Wikipedia lookup failed: {type(e).__name__}: {str(e)}"
    
#  Tool 3: Calculator
@tool
def calculator(expression: str) -> str:
    """ Calculate the result of a mathematical expression. Use this for math problems, unit conversions, or any calculation.
    Arguments:
        expression (str): A mathematical expression to evaluate, e.g. "2 + 2", "5 * (3 - 1)", "100 USD to EUR".
    Returns:
        str: The result of the calculation.
    """
    try:
        import numexpr # import lazyly to keep module load fast
        
        result = numexpr.evaluate(expression).item()  # evaluate expression and get scalar result
        return str(result)
    except Exception as e:
        return f"Calculation failed on {expression}: {type(e).__name__}: {str(e)}"
    
# A flat list of all tools for easy import
ALL_TOOLS = [web_search, wikipedia_lookup, calculator]

if __name__ == "__main__":
    # Quick test of tools
    print("Testing tools...")
    print("\nWeb Search for 'Anthropic Claude 4 release':")
    print(web_search.invoke("Anthropic Claude 4 release"))
    
    print("\nWikipedia lookup for 'Apache Spark':")
    print(wikipedia_lookup.invoke("Apache Spark"))
    
    
