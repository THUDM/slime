# Slime Router Tests

Comprehensive test suite for the Slime Router's OpenAI Chat Completion API with radix cache integration.

## Status

- **Tests**: 75/75 passing (100%)
- **Coverage**: ~98%+
- **Last Updated**: 2025-11-26

## Test Structure

```
tests/router/
├── conftest.py                          # Shared pytest fixtures + helpers
├── test_chat_validation.py              # Request validation (10 tests)
├── test_openai_format.py                # OpenAI format compatibility (13 tests)
├── test_radix_cache_integration.py      # Radix cache integration (19 tests)
├── test_openai_chat_completion.py       # Error handling & sampling (22 tests)
├── test_parsers.py                      # Parser integration (11 tests)
└── fixtures/
    ├── mock_responses.py                # Mock SGLang responses
    └── sample_requests.py               # Sample OpenAI request data
```

## Quick Start

### Install dependencies

```bash
pip install pytest pytest-asyncio pytest-mock pytest-cov respx
```

### Run all tests

```bash
pytest tests/router/ -v
```

### Run with coverage

```bash
pytest tests/router/ --cov=slime.router.handlers.openai_chat_completion --cov-report=term-missing
```

## Test Categories

### P0 - Critical (57 tests)

| Category | Tests | Description |
|----------|-------|-------------|
| Request Validation | 10 | Input validation, message structure, null content handling |
| OpenAI Format | 13 | Response format, required fields, tool_calls, finish_reason variations |
| Radix Cache | 19 | Token retrieval/insertion, multi-rollout, cache fallback, edge cases |
| Error Handling | 7 | SGLang errors (400/500/429), timeouts, connections, malformed JSON |
| Handler Init | 3 | Lazy loading, property caching |
| End-to-End | 2 | Complete request flow with/without cache |
| Response Parsing | 2 | SGLang response edge cases, missing logprobs |
| Resource Cleanup | 2 | URL cleanup on timeout/error |

### P1 - Important (18 tests)

| Category | Tests | Description |
|----------|-------|-------------|
| Sampling Parameters | 5 | Parameter defaults, stop conditions, None removal |
| Parser Integration | 11 | Reasoning/function parsers, combinations (optional, requires SGLang) |
| Generation | 4 | Cache maintenance, worker URL management |
| Direct Proxy | 2 | Proxy mode to SGLang |

## Recent Improvements

### Test Suite Enhancements (2025-11-26)

Added **11 new tests** to improve coverage from 95% → 98%:

**New Test Coverage:**
- End-to-end integration tests (2)
- SGLang response parsing edge cases (2)
- finish_reason format variations (4 via parametrization)
- Parser combinations (1)
- Cache retrieval edge cases (1)
- Resource cleanup on exceptions (2)

**Code Quality Improvements:**
- Created `mock_router_with_components` fixture (eliminates 15+ duplications)
- Created `mock_generate_endpoint()` helper function (eliminates 8+ duplications)
- Parametrized error tests: 3 tests → 1 parametrized test (58% code reduction)
- Consolidated documentation: 3 files (768 lines) → 1 file (109 lines)

## Key Fixes Applied

| Issue | Impact | Solution | Tests Fixed |
|-------|--------|----------|-------------|
| respx mock config | 22 tests | Use real httpx.AsyncClient | 22 |
| Non-existent method | Radix cache broken | Use retrieve_from_text() | N/A |
| finish_reason format | 1 test | Handle dict and string | 1 |
| Logprobs mismatch | 2 tests | Combine input+output logprobs | 2 |
| Mock reassignment | 1 test | Use side_effect | 1 |

## Mocking Strategy

**Core Mocks:**
- **Router**: Full mock with component registry
- **RadixTree**: Mock retrieve_from_text() and insert()
- **Tokenizer**: Mock apply_chat_template(), encode(), decode()
- **HTTP**: respx library for httpx mocking

**Reusable Fixtures:**
- `mock_router_with_components` - Router with pre-configured radix_tree and tokenizer
- `mock_generate_endpoint()` - Helper to mock SGLang /generate endpoint with respx

**Best Practices:**
- Use `mock_router_with_components` when tests need both radix_tree and tokenizer
- Use `mock_generate_endpoint()` to quickly mock SGLang responses
- Use parametrized tests (`@pytest.mark.parametrize`) for similar test cases
- All async tests must use `@pytest.mark.asyncio` decorator

## Troubleshooting

### Async test errors

```bash
pip install pytest-asyncio
```

### Parser tests skipped

SGLang parsers are optional. Tests automatically skip if not installed.

### Mock import errors

```bash
pip install pytest pytest-asyncio pytest-mock respx httpx fastapi
```

## Contributing

When adding tests:

1. **Use reusable fixtures** - Use `mock_router_with_components` and `mock_generate_endpoint()` to reduce duplication
2. **Parametrize similar tests** - Use `@pytest.mark.parametrize` instead of writing multiple similar tests
3. **Follow naming conventions** - Test classes start with `Test`, test functions start with `test_`
4. **Write clear docstrings** - Explain what scenario the test covers and why
5. **Use `@pytest.mark.asyncio`** - Required for all async test functions
6. **Update this README** - Update test counts and categories when adding new tests

**Example of good test:**

```python
@pytest.mark.parametrize("input,expected", [
    ({"type": "stop"}, "stop"),
    ("stop", "stop"),
])
@pytest.mark.asyncio
@respx.mock
async def test_finish_reason_extraction(
    self, mock_router_with_components, mock_radix_tree, mock_tokenizer,
    input, expected
):
    """Test finish_reason extraction handles both dict and string formats."""
    # Test implementation
```
