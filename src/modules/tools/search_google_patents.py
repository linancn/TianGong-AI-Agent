from typing import Optional, Type

from google.cloud import bigquery
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool
from pydantic import BaseModel


class SearchGooglePatents(BaseTool):
    name = "search_google_patents"
    description = "Search patents through google big query."

    class InputSchema(BaseModel):
        query: str

    args_schema: Type[BaseModel] = InputSchema

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool synchronously."""
        client = bigquery.Client()
        # Construct a BigQuery client object.

        query = """
            SELECT name, SUM(number) as total_people
            FROM `bigquery-public-data.usa_names.usa_1910_2013`
            WHERE state = 'TX'
            GROUP BY name, state
            ORDER BY total_people DESC
            LIMIT 20
        """
        query_job = client.query(query)  # Make an API request.

        print("The query data:")
        for row in query_job:
            # Row values can be accessed by field name or index.
            print("name={}, count={}".format(row[0], row["total_people"]))

        return docs_list

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
