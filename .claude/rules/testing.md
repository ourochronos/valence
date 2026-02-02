# Valence Testing Rules

## Test Requirements

### Bug Fixes Require Tests

When fixing a bug:
1. **Write tests first** that reproduce the bug (they should fail)
2. **Implement the fix** that makes the tests pass
3. **Add regression tests** to prevent the bug from returning

A bug fix without tests is incomplete. Tests document the expected behavior and prevent regressions.

### Test Patterns

- **Unit tests**: Test individual functions/methods in isolation
- **Integration tests**: Test components working together (e.g., database interactions)
- **Regression tests**: Specifically test scenarios that previously caused bugs

### Test Location

- Unit tests: `tests/<module>/test_<filename>.py`
- Integration tests: `tests/integration/test_<feature>.py`
- Test fixtures: `tests/conftest.py` for shared fixtures

### Test Naming

```python
def test_<action>_<scenario>():
    """Should <expected behavior>."""
    pass

# Examples:
def test_creates_new_session_when_none_exists():
def test_resumes_existing_session_on_restart():
def test_handles_database_error_gracefully():
```

### Mocking Guidelines

- Mock external services (database, APIs, filesystem)
- Use `pytest` fixtures for reusable mocks
- Don't mock the code under test
- Prefer `unittest.mock.patch` as context managers

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/agents/test_matrix_bot.py

# Run specific test class
pytest tests/agents/test_matrix_bot.py::TestGetOrCreateSession

# Run with coverage
pytest --cov=valence

# Run in verbose mode
pytest -v
```

## Test Coverage Goals

- New code should have tests
- Bug fixes must have tests
- Critical paths (authentication, data persistence) should have thorough coverage

## Async Testing

For async code:
```python
@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result == expected
```

## Database Testing

- Use `mock_get_cursor` fixture for unit tests
- Use transaction rollback for integration tests
- Never run tests against production database
