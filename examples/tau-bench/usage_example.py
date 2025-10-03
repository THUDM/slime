"""
OpenAI Compatible Tool Call Adapter Usage Example

This file demonstrates how to integrate OpenAICompatibleToolCallAdapter
into the existing trainable_agents.py
"""

from openai_tool_adapter import create_openai_adapter
from trainable_agents import TrainableToolCallingAgent


def demonstrate_usage():
    """
    Demonstrate how to use OpenAI compatible adapter
    """
    # Example tool information
    tools_info = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search web information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string",
                                  "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    # Create adapter directly
    adapter = create_openai_adapter(tools_info, "qwen25")

    # Example response
    sample_response = """
    I need to search for some information.

    <tool_call>
    <tool_name>search_web</tool_name>
    <parameters>{"query": "latest AI developments"}</parameters>
    </tool_call>
    """

    # Parse to OpenAI format
    print("\n=== Testing OpenAI Adapter ===")
    result = adapter.parse_response_to_openai_format(sample_response)

    if result["success"]:
        print("✅ Parsing successful!")
        print(f"OpenAI message: {result['openai_message']}")

        # Get OpenAI format tool definitions
        print("\n=== Getting OpenAI Tools Format ===")
        openai_tools = adapter.get_openai_tools_format()
        print(f"OpenAI tool definitions: {openai_tools}")

    else:
        print(f"❌ Parsing failed: {result['error']}")


def demonstrate_integration():
    """
    Demonstrate how the adapter is integrated in trainable_agents.py
    """
    # Example tool information
    tools_info = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search web information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string",
                                  "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    # Create agent with OpenAI adapter integrated
    agent = TrainableToolCallingAgent(
        tools_info=tools_info,
        wiki="You are a helpful assistant.",
        model="qwen2.5-3b",
        provider="local"
    )

    # The agent now has OpenAI adapter methods available:
    print("\n=== Available OpenAI Methods ===")
    print("✅ get_openai_tools_format()")
    print("✅ get_openai_message_from_response(response)")
    
    # Get OpenAI format tools
    print("\n=== Testing Agent OpenAI Methods ===")
    openai_tools = agent.get_openai_tools_format()
    print(f"OpenAI tools: {openai_tools}")

    # Example of parsing a response
    sample_response = """
    I need to search for information.

    <tool_call>
    <tool_name>search_web</tool_name>
    <parameters>{"query": "machine learning"}</parameters>
    </tool_call>
    """

    print("\n=== Testing Agent Response Parsing ===")
    openai_result = agent.get_openai_message_from_response(sample_response)
    if openai_result["success"]:
        print(f"✅ OpenAI format message: {openai_result['openai_message']}")
    else:
        print(f"❌ Failed to parse: {openai_result['error']}")


if __name__ == "__main__":
    print("=== Direct Adapter Usage ===")
    demonstrate_usage()
    
    print("\n=== Integration in TrainableAgent ===")
    demonstrate_integration()
