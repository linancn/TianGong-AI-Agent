import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from modules.tools.search_google_patents_tool import SearchGooglePatents

search_google_patents = SearchGooglePatents()

query = "energy"

patents_docs = search_google_patents._run(query=query)
