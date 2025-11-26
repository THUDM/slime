"""Shared pytest fixtures for router tests."""

import pytest
from unittest.mock import Mock, AsyncMock
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def mock_router():
    """Mock SlimeRouter with all necessary attributes."""
    router = Mock()

    # Mock args
    router.args = Mock()
    router.args.verbose = False
    router.args.sglang_reasoning_parser = None
    router.args.sglang_tool_call_parser = None
    router.args.slime_router_generation_timeout = 120.0
    router.args.model_name = "test-model"
    router.args.include_reasoning_in_response = False
    router.args.hf_checkpoint = "test-checkpoint"

    # Mock component registry
    router.component_registry = Mock()
    router.component_registry.get = Mock()
    router.component_registry.has = Mock(return_value=True)

    # Mock HTTP client (httpx) - use real AsyncClient for respx to work
    import httpx
    router.client = httpx.AsyncClient()
    # respx will intercept calls from this real client in tests

    # Mock worker URL management
    router._use_url = AsyncMock(return_value="http://localhost:30000")
    router._finish_url = AsyncMock()
    router._check_cache_availability = Mock(return_value=True)

    return router


@pytest.fixture
def mock_radix_tree():
    """Mock StringRadixTrie for cache operations."""
    radix_tree = Mock()

    # Mock retrieve_from_text (existing synchronous method)
    radix_tree.retrieve_from_text = Mock(
        return_value=([1, 2, 3, 4], [0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0])
    )

    # Mock insert method
    radix_tree.insert = Mock(return_value=True)

    # Mock find_longest_prefix
    from slime.router.core.radix_tree import MatchResult
    radix_tree.find_longest_prefix = Mock(
        return_value=MatchResult(
            matched_prefix="",
            token_ids=[],
            logp=[],
            loss_mask=[],
            remaining_string="test input",
            last_node=Mock()
        )
    )

    return radix_tree


@pytest.fixture
def mock_tokenizer():
    """Mock HuggingFace tokenizer."""
    tokenizer = Mock()

    # Mock apply_chat_template
    tokenizer.apply_chat_template = Mock(
        return_value="<|system|>You are helpful<|user|>Hello<|assistant|>"
    )

    # Mock encode
    tokenizer.encode = Mock(return_value=[1, 2, 3, 4])

    # Mock decode
    tokenizer.decode = Mock(return_value="Generated response")

    # Mock callable for tokenization
    tokenizer.__call__ = Mock(return_value={"input_ids": [1, 2, 3, 4]})

    return tokenizer


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]


@pytest.fixture
def sample_request_data(sample_messages):
    """Sample request data."""
    return {
        "messages": sample_messages,
        "max_tokens": 100,
        "temperature": 0.7
    }


@pytest.fixture
async def mock_request(sample_request_data):
    """Mock FastAPI Request object."""
    async def mock_json():
        return sample_request_data

    request = Mock()
    request.json = mock_json
    return request


@pytest.fixture
def mock_reasoning_parser():
    """Mock SGLang ReasoningParser."""
    # Check if parsers are available
    try:
        from slime.router.handlers.openai_chat_completion import SGLANG_PARSERS_AVAILABLE
        if not SGLANG_PARSERS_AVAILABLE:
            pytest.skip("SGLang parsers not available")
    except ImportError:
        pytest.skip("SGLang parsers not available")

    parser = Mock()
    parser.parse_non_stream = Mock(
        return_value=("Reasoning content", "Normal text")
    )
    return parser


@pytest.fixture
def mock_function_call_parser():
    """Mock SGLang FunctionCallParser."""
    # Check if parsers are available
    try:
        from slime.router.handlers.openai_chat_completion import SGLANG_PARSERS_AVAILABLE
        if not SGLANG_PARSERS_AVAILABLE:
            pytest.skip("SGLang parsers not available")
    except ImportError:
        pytest.skip("SGLang parsers not available")

    parser = Mock()

    # Mock ToolCallItem
    tool_call_item = Mock()
    tool_call_item.name = "get_weather"
    tool_call_item.arguments = {"location": "San Francisco"}

    parser.parse_non_stream = Mock(
        return_value=("", [tool_call_item])  # (remaining_text, tool_calls)
    )
    return parser
