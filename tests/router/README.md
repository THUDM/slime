# Slime Router Tests

Comprehensive test suite for the Slime Router's OpenAI Chat Completion API with radix cache integration.

## Test Structure

```
tests/router/
├── conftest.py                          # Shared pytest fixtures
├── test_chat_validation.py              # Request validation tests (P0)
├── test_openai_format.py                # OpenAI format compatibility tests (P0)
├── test_radix_cache_integration.py      # Radix cache integration tests (P0)
├── test_openai_chat_completion.py       # Error handling & sampling tests (P0/P1)
├── test_parsers.py                      # Parser integration tests (P1, optional)
└── fixtures/
    ├── mock_responses.py                # Mock SGLang responses
    └── sample_requests.py               # Sample OpenAI request data
```

## Prerequisites

Install test dependencies:

```bash
pip install pytest pytest-asyncio pytest-mock pytest-cov respx
```

Or use the project's test dependencies (when added to pyproject.toml):

```bash
pip install -e ".[test]"
```

## Running Tests

### Run all router tests
```bash
pytest tests/router/ -v
```

### Run specific test file
```bash
pytest tests/router/test_chat_validation.py -v
```

### Run with coverage report
```bash
pytest tests/router/ --cov=slime.router.handlers.openai_chat_completion --cov-report=html --cov-report=term-missing
```

### Run only P0 (critical) tests
```bash
# Validation tests
pytest tests/router/test_chat_validation.py -v

# Format compatibility tests
pytest tests/router/test_openai_format.py -v

# Radix cache integration tests
pytest tests/router/test_radix_cache_integration.py -v

# Error handling tests
pytest tests/router/test_openai_chat_completion.py::TestErrorHandling -v
```

### Skip parser tests (if SGLang parsers not installed)
```bash
pytest tests/router/ -v -k "not parser"
```

### Run tests with verbose output
```bash
pytest tests/router/ -vv --tb=short
```

## Test Categories

### P0 - Critical Tests (Must Pass)

1. **Request Validation** (`test_chat_validation.py`)
   - Tests input validation for chat completion requests
   - Ensures invalid requests are properly rejected
   - Validates message structure and content handling

2. **OpenAI Format Compatibility** (`test_openai_format.py`)
   - Verifies response format matches OpenAI API spec
   - Tests all required fields (id, choices, usage, etc.)
   - Validates finish_reason and tool_calls format

3. **Radix Cache Integration** (`test_radix_cache_integration.py`)
   - Tests token retrieval from radix cache
   - Verifies cache insertion after generation
   - Tests multi-rollout scenarios
   - Validates cache fallback behavior

4. **Error Handling** (`test_openai_chat_completion.py::TestErrorHandling`)
   - Tests SGLang error mapping (400, 500, 429)
   - Tests connection and timeout errors
   - Validates error response format

### P1 - Important Tests

1. **Sampling Parameters** (`test_openai_chat_completion.py::TestSamplingParams`)
   - Tests parameter building and defaults
   - Validates stop conditions
   - Tests None value removal

2. **Parser Integration** (`test_parsers.py`)
   - Tests reasoning parser (optional, requires SGLang)
   - Tests function call parser (optional, requires SGLang)
   - Tests combined parser usage
   - Skipped automatically if parsers not available

## Key Test Scenarios

### Multi-Rollout Testing

The test suite includes comprehensive multi-rollout scenarios:

```python
# Test that multiple generations correctly store tokens/logprobs
test_multiple_rollouts_insert_called_multiple_times
test_multiple_rollouts_logprobs_match_tokens
test_cached_tokens_reused_in_subsequent_requests
```

### Mocking Strategy

- **Router**: Mocked with all necessary attributes and methods
- **RadixTree**: Mocked `retrieve_from_text()` and `insert()` methods
- **Tokenizer**: Mocked `apply_chat_template()`, `encode()`, `decode()`
- **HTTP Client**: Uses `respx` library for httpx mocking

## Coverage Goals

- Overall code coverage: 90%+
- ChatCompletionHandler core methods: 95%+
- Request validation: 100%
- Error handling: 100%

## Current Coverage

Run coverage report:

```bash
pytest tests/router/ --cov=slime.router.handlers.openai_chat_completion --cov-report=term-missing
```

View HTML coverage report:

```bash
pytest tests/router/ --cov=slime.router.handlers.openai_chat_completion --cov-report=html
open htmlcov/index.html
```

## Troubleshooting

### Async test errors

If you see "async def functions are not natively supported":

```bash
pip install pytest-asyncio
```

### Parser tests skipped

Parser tests require SGLang installation. If SGLang is not installed, these tests will be automatically skipped. This is expected behavior.

### Mock import errors

Ensure all dependencies are installed:

```bash
pip install pytest pytest-asyncio pytest-mock respx httpx fastapi
```

## Contributing

When adding new tests:

1. Follow the existing test structure and naming conventions
2. Use appropriate fixtures from `conftest.py`
3. Mark tests with proper priority (P0/P1)
4. Add docstrings explaining what the test validates
5. Use `@pytest.mark.asyncio` for async tests
6. Use `@pytest.mark.skipif` for optional dependency tests
7. Update this README with new test categories

## Test Maintenance

- Run tests before committing changes
- Maintain 90%+ coverage for all new code
- Update mocks when interfaces change
- Keep test data in fixtures/ directory
- Document any new testing patterns
