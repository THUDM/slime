import asyncio
import os
import random
import re

import aiohttp
import chardet


# --- Utilities ---
def parse_snippet(snippet: str) -> list[str]:
    segments = snippet.split("...")
    return [s.strip() for s in segments if len(s.strip().split()) > 5]


def sanitize_search_query(query: str) -> str:
    # Remove or replace special characters that might cause issues.
    # This is a basic example; you might need to add more characters or patterns.
    sanitized_query = re.sub(r"[^\w\s]", " ", query)  # Replace non-alphanumeric and non-whitespace with spaces.
    sanitized_query = re.sub(
        r"[\t\r\f\v\n]", " ", sanitized_query
    )  # replace tab, return, formfeed, vertical tab with spaces.
    sanitized_query = re.sub(
        r"\s+", " ", sanitized_query
    ).strip()  # remove duplicate spaces, and trailing/leading spaces.

    return sanitized_query


def filter_links(search_results: list[dict]) -> list[str]:
    links = []
    for result in search_results:
        for item in result.get("items", []):
            if "mime" in item:
                continue
            ext = os.path.splitext(item["link"])[1]
            if ext in ["", ".html", ".htm", ".shtml"]:
                links.append(item["link"])
    return links


async def fetch(session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore) -> str:
    if url == "":
        return ""
    user_agents = [
        "Mozilla/5.0 (Linux; Android 6.0.1; Nexus 5X Build/MMB29P)...",
        "Mozilla/5.0 AppleWebKit/537.36...",
        "Mozilla/5.0 (compatible; Googlebot/2.1; +https://www.google.com/bot.html)",
    ]
    headers = {"User-Agent": random.choice(user_agents)}

    async with semaphore:
        try:
            async with session.get(url, headers=headers) as response:
                raw = await response.read()
                detected = chardet.detect(raw)
                encoding = detected["encoding"] or "utf-8"
                return raw.decode(encoding, errors="ignore")
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return ""


async def fetch_all(urls: list[str], limit: int = 8) -> list[str]:
    semaphore = asyncio.Semaphore(limit)
    timeout = aiohttp.ClientTimeout(total=5)
    connector = aiohttp.TCPConnector(limit_per_host=limit, force_close=True)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [fetch(session, url, semaphore) for url in urls]
        return await asyncio.gather(*tasks)


def collect_context(snippet: str, doc: str) -> str:
    snippets = parse_snippet(snippet)
    ctx_paras = []

    for s in snippets:
        pos = doc.replace("\n", " ").find(s)
        if pos == -1:
            continue
        sta = pos
        while sta > 0 and doc[sta] != "\n":
            sta -= 1
        end = pos + len(s)
        while end < len(doc) and doc[end] != "\n":
            end += 1
        para = doc[sta:end].strip()
        if para not in ctx_paras:
            ctx_paras.append(para)

    return "\n".join(ctx_paras)


async def google_search(api_key, query, top_k=5, timeout: int = 60, proxy=None, snippet_only=False) -> list[dict]:
    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    session_kwargs = {}
    if proxy:
        session_kwargs["proxy"] = proxy
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(
            "https://google.serper.dev/search",
            json={
                "q": query,
                "num": top_k,
                "gl": "us",
                "hl": "en",
            },
            headers={
                "Content-Type": "application/json",
                "X-API-KEY": api_key,
            },
            timeout=timeout_obj,
        ) as resp:
            resp.raise_for_status()
            response = await resp.json()
            items = response.get("organic", [])

    contexts = []
    if snippet_only:
        for item in items:
            title = item.get("title", "")
            context = " ".join(parse_snippet(item.get("snippet", "")))
            if title != "" or context != "":
                title = "No title." if not title else title
                context = "No snippet available." if not context else context
                contexts.append(
                    {
                        "document": {"contents": f'"{title}"\n{context}'},
                    }
                )
    else:
        links = [item.get("link", "") for item in items if "link" in item]
        web_contents = await fetch_all(links)
        contexts = []
        for i, item in enumerate(items):
            title = item.get("title", "")
            snippet = item.get("snippet", "")

            context = collect_context(snippet, web_contents[i])
            if title != "" or context != "":
                title = "No title." if not title else title
                context = "No snippet available." if not context else context
                contexts.append(
                    {
                        "document": {"contents": f'"{title}"\n{context}'},
                    }
                )

    return contexts
