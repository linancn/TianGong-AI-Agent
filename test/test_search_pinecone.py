import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.modules.tools.common.search_pinecone import SearchPinecone

search_pinecone = SearchPinecone()

k = 3
query = "what is dynamic material flow analysis?"
extend = 1

pinecone_docs = search_pinecone.sync_similarity(query=query, top_k=k, extend=extend)
