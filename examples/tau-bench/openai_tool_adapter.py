from typing import Dict, Any, List, Optional
import json
import logging
from dataclasses import dataclass
from sglang_tool_parser import parse_tools
from tau_bench.types import Action
from tau_bench.agents.tool_calling_agent import RESPOND_ACTION_NAME

# Set up logger for this module
logger = logging.getLogger(__name__)


@dataclass
class OpenAIToolCall:
    """OpenAI format tool call structure"""
    id: str
    type: str = "function"
    function: Dict[str, Any] = None


@dataclass
class OpenAIAssistantMessage:
    """OpenAI format assistant message structure"""
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None


class OpenAICompatibleToolCallAdapter:
    """
    Adapter class that converts sglang tool call parsing results to OpenAI compatible format

    This class encapsulates existing tool call parsing and action conversion logic,
    and provides OpenAI format output interface.
    """

    def __init__(self, tools_info: List[Dict[str, Any]],
                 parser_type: str = "qwen25"):
        """
        Initialize adapter

        Args:
            tools_info: List of tool information
            parser_type: Parser type, defaults to "qwen25"
        """
        self.tools_info = tools_info
        self.parser_type = parser_type

    def parse_response_to_openai_format(
        self,
        response: str
    ) -> Dict[str, Any]:
        """
        Parse sglang response to OpenAI compatible format

        Args:
            response: Raw response text from sglang

        Returns:
            Dictionary containing OpenAI format message and parsing results

        Raises:
            Exception: Thrown when parsing fails
        """
        logger.debug(f"Starting to parse response: {response[:100]}...")
        
        try:
            # Use existing parser to parse tool calls
            logger.debug(f"Using parser type: {self.parser_type}")
            parsed = parse_tools(response, self.tools_info, self.parser_type)
            logger.debug(f"Parsing successful. Normal text: '{parsed['normal_text']}'")
            logger.debug(f"Found {len(parsed['calls'])} tool calls: {parsed['calls']}")

            # Extract parsing results
            normal_text = parsed['normal_text']
            calls = parsed['calls']

            # Convert to OpenAI format
            openai_message = self._convert_to_openai_message(
                normal_text, calls)
            logger.debug(f"OpenAI message created: {openai_message}")

            return {
                "openai_message": openai_message,
                "parsed_result": parsed,
                "success": True
            }

        except Exception as e:
            logger.warning(f"Parsing failed with error: {str(e)}")
            return {
                "openai_message": None,
                "parsed_result": None,
                "success": False,
                "error": str(e)
            }

    def _convert_to_openai_message(
        self,
        normal_text: str,
        calls: List[Dict[str, Any]]
    ) -> OpenAIAssistantMessage:
        """
        Convert parsing results to OpenAI format assistant message

        Args:
            normal_text: Normal text content
            calls: List of tool calls

        Returns:
            OpenAI format assistant message
        """
        logger.debug(f"Converting to OpenAI format - normal_text: '{normal_text}', calls: {calls}")
        
        if not calls:
            # No tool calls, return plain text response
            logger.debug("No tool calls found, returning plain text response")
            return OpenAIAssistantMessage(
                role="assistant",
                content=normal_text,
                tool_calls=None
            )

        # Convert tool calls to OpenAI format
        logger.debug(f"Converting {len(calls)} tool calls to OpenAI format")
        openai_tool_calls = []
        for i, call in enumerate(calls):
            logger.debug(f"Processing call {i}: {call}")
            openai_tool_call = OpenAIToolCall(
                id=f"call_{i}_{call.get('name', 'unknown')}",
                type="function",
                function={
                    "name": call.get('name', ''),
                    "arguments": call.get('parameters', '{}')
                }
            )
            logger.debug(f"Created OpenAI tool call: {openai_tool_call}")
            openai_tool_calls.append(openai_tool_call)

        result = OpenAIAssistantMessage(
            role="assistant",
            content=normal_text if normal_text.strip() else None,
            tool_calls=openai_tool_calls
        )
        logger.debug(f"Final OpenAI message: {result}")
        return result

    def convert_to_action(
        self,
        response: str
    ) -> Action:
        """
        Convert response to Action object (maintain compatibility with existing logic)

        Args:
            response: Raw response text from sglang

        Returns:
            Action object
        """
        logger.debug(f"Converting response to Action: {response[:100]}...")
        
        try:
            parsed = parse_tools(response, self.tools_info, self.parser_type)
            agent_content = parsed['normal_text']
            calls = parsed['calls']
            logger.debug(f"Parsed for Action - content: '{agent_content}', calls: {calls}")

            action = self._call_to_action_sglang(calls, agent_content)
            logger.debug(f"Created Action: {action}")
            return action

        except Exception as e:
            logger.warning(f"Failed to convert response to action: {e}")
            # Return default response action
            default_action = Action(name=RESPOND_ACTION_NAME,
                          kwargs={"content": response})
            logger.debug(f"Returning default action: {default_action}")
            return default_action

    def _call_to_action_sglang(self, calls: List[Any],
                               text_response: str) -> Action:
        """
        Convert sglang tool calls to Action object

        This method replicates the original call_to_action_sglang logic,
        ensuring compatibility with existing code.
        """
        logger.debug(f"Converting sglang calls to Action - calls: {calls}, text: '{text_response}'")
        
        # Default action if no action found
        action = Action(name=RESPOND_ACTION_NAME,
                        kwargs={"content": text_response})
        logger.debug(f"Default action created: {action}")

        if calls:
            logger.debug(f"Processing {len(calls)} tool calls")
            if len(calls) > 1:
                logger.debug("Multiple tool calls identified, only taking first.")

            tool_call = calls[0]
            logger.debug(f"Processing tool call: {tool_call}")
            
            try:
                params = json.loads(tool_call["parameters"])
                logger.debug(f"Parsed parameters: {params}")

                if not isinstance(params, dict):
                    logger.warning(f"{params} does not follow dict structure for action")
                else:
                    action = Action(
                        name=tool_call["name"],
                        kwargs=params
                    )
                    logger.debug(f"Created tool action: {action}")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse parameters as JSON: {e}")
        else:
            logger.debug("No tool calls found, using default action")

        logger.debug(f"Final action: {action}")
        return action

    def get_openai_tools_format(self) -> List[Dict[str, Any]]:
        """
        Get OpenAI format tool definitions

        Returns:
            List of OpenAI format tools
        """
        openai_tools = []
        for tool in self.tools_info:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool['function']['name'],
                    "description": tool['function']['description'],
                    "parameters": tool['function']['parameters']
                }
            }
            openai_tools.append(openai_tool)

        return openai_tools

    def validate_response(self, response: str) -> bool:
        """
        Validate if response can be parsed correctly

        Args:
            response: Response text to validate

        Returns:
            Whether it can be parsed correctly
        """
        try:
            parse_tools(response, self.tools_info, self.parser_type)
            return True
        except Exception:
            return False


# Usage examples and factory functions
def create_openai_adapter(
    tools_info: List[Dict[str, Any]],
    parser_type: str = "qwen25"
) -> OpenAICompatibleToolCallAdapter:
    """
    Factory function to create OpenAI compatible tool call adapter

    Args:
        tools_info: List of tool information
        parser_type: Parser type

    Returns:
        Configured adapter instance
    """
    return OpenAICompatibleToolCallAdapter(tools_info, parser_type)
