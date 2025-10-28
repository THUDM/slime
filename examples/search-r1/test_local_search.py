#!/usr/bin/env python3
"""
Test script for Local Search Server

This script demonstrates how to use local_search_server.py to perform a search query
using a local retrieval engine.

REQUIREMENTS:
1. You must have a local retrieval server running first:

   cd /root/Search-R1
   conda activate retriever

   # For BM25:
   python search_r1/search/retrieval_server.py \
       --index_path /path/to/bm25 \
       --corpus_path /path/to/wiki-18.jsonl \
       --topk 3 \
       --retriever_name bm25

   # Or for E5 dense retriever:
   python search_r1/search/retrieval_server.py \
       --index_path /path/to/e5_Flat.index \
       --corpus_path /path/to/wiki-18.jsonl \
       --topk 3 \
       --retriever_name e5 \
       --retriever_model intfloat/e5-base-v2 \
       --faiss_gpu

2. The server should be running on http://127.0.0.1:8000 (default)
"""

import asyncio
from local_search_server import local_search


async def main():
    # Local retrieval server URL
    search_url = "http://127.0.0.1:8000/retrieve"

    # Search query (same as Google search test)
    query = "Amazon"

    # Number of results to retrieve
    top_k = 5

    print(f"Searching for: '{query}'")
    print(f"Using local retrieval server at: {search_url}")
    print(f"Retrieving top {top_k} results...\n")
    print("=" * 80)

    try:
        # Perform the search
        results = await local_search(
            search_url=search_url,
            query=query,
            top_k=top_k,
            timeout=60,
            proxy=None,
            snippet_only=False  # Not used for local search
        )

        # Print results
        if not results:
            print("No results found.")
            print("\nPossible issues:")
            print("1. Local retrieval server is not running")
            print("2. Query has no matching documents in the corpus")
            print("3. Server URL is incorrect")
            print(f"\nTry checking if the server is running:")
            print(f"  curl {search_url} -X POST -H 'Content-Type: application/json' -d '{{\"queries\": [\"test\"], \"topk\": 3}}'")
        else:
            print(results)

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

        print("\n" + "=" * 80)
        print("TROUBLESHOOTING:")
        print("=" * 80)
        print("Make sure your local retrieval server is running!")
        print("\nTo start the server, run:")
        print("  cd /root/Search-R1")
        print("  conda activate retriever")
        print("  python search_r1/search/retrieval_server.py --index_path <index> --corpus_path <corpus> --topk 3")


if __name__ == "__main__":
    asyncio.run(main())
