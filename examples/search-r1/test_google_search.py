#!/usr/bin/env python3
"""
Test script for Google Search Server

This script demonstrates how to use google_search_server.py to perform a search query.
"""

import asyncio
from google_search_server import google_search


async def main():
    # Your Serper API key
    api_key = "6f32c0e10507ce84f69751cda90614374bccf35a"

    # Search query
    query = "when Amazon is created"

    # Number of results to retrieve
    top_k = 5

    print(f"Searching for: '{query}'")
    print(f"Retrieving top {top_k} results...\n")
    print("=" * 80)

    try:
        # Perform the search
        results = await google_search(
            api_key=api_key,
            query=query,
            top_k=top_k,
            snippet_only=True,  # Set to True to only get snippets (faster)
            proxy=None
        )

        print(results)
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
