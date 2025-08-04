# from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
# from langchain_community.tools import DuckDuckGoSearchResults

# wrapper = DuckDuckGoSearchAPIWrapper(region="in-en", source="text", time="d", max_results=20)

# search = DuckDuckGoSearchResults(api_wrapper=wrapper)

# print(search.invoke("earphones under 1000 Rs prices India 2024"))

from langchain_community.utilities import SearxSearchWrapper
search = SearxSearchWrapper(searx_host="http://127.0.0.1:8080", unsecure=True)
print(search.run("What is the capital of France"))