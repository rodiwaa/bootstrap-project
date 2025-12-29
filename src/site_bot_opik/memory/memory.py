from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from dotenv import load_dotenv
import os
from urllib.parse import urlparse

load_dotenv()

def init_llm_cache():
  print('init_llm_cache')
  
  set_llm_cache(
    InMemoryCache()
  )
