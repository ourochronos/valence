"""Tests for WU-16: Inference task schemas and validate_output (DR-11).

Covers:
- Valid output parses correctly for each task type
- Missing required fields raise InferenceSchemaError
- Invalid enum values raise InferenceSchemaError
- Markdown fence stripping works
- Extra fields are ignored (not an error)
- Default values applied for optional fields

asyncio_mode = auto (pyproject.toml), no @pytest.mark.asyncio needed.
"""

from __future__ import annotations

import json

import pytest

from valence.core.inference import (
    RELATIONSHIP_ENUM,
    TASK_CLASSIFY,
    TASK_COMPILE,
    TASK_CONTENTION,
    TASK_OUTPUT_SCHEMAS,
    TASK_SPLIT,
    TASK_UPDATE,
    InferenceProvider,
    InferenceSchemaError,
    provider,
    validate_output,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_provider():
    """Reset module-level singleton after each test."""
    yield
    provider.configure(None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _j(obj: dict) -> str:
    return json.dumps(obj)


# ---------------------------------------------------------------------------
# RELATIONSHIP_ENUM
# ---------------------------------------------------------------------------


class TestRelationshipEnum:
    def test_all_values_present(self):
        assert set(RELATIONSHIP_ENUM) == {"originates", "confirms", "supersedes", "contradicts", "contends"}

    def test_is_list(self):
        assert isinstance(RELATIONSHIP_ENUM, list)


# ---------------------------------------------------------------------------
# TASK_OUTPUT_SCHEMAS
# ---------------------------------------------------------------------------


class TestTaskOutputSchemas:
    def test_all_task_types_have_schema(self):
        for task in (TASK_COMPILE, TASK_UPDATE, TASK_CLASSIFY, TASK_CONTENTION, TASK_SPLIT):
            assert task in TASK_OUTPUT_SCHEMAS
            assert isinstance(TASK_OUTPUT_SCHEMAS[task], str)
            assert len(TASK_OUTPUT_SCHEMAS[task]) > 0


# ---------------------------------------------------------------------------
# validate_output — TASK_COMPILE
# ---------------------------------------------------------------------------


_UNSET = object()


class TestValidateCompile:
    def _valid(self, source_relationships=_UNSET) -> dict:
        if source_relationships is _UNSET:
            source_relationships = [{"source_id": "abc-123", "relationship": "originates"}]
        return {
            "title": "Test Article",
            "content": "Some content.",
            "source_relationships": source_relationships,
        }

    def test_valid_output_returns_dict(self):
        result = validate_output(TASK_COMPILE, _j(self._valid()))
        assert result["title"] == "Test Article"
        assert result["content"] == "Some content."
        assert len(result["source_relationships"]) == 1

    def test_missing_title_raises(self):
        data = self._valid()
        del data["title"]
        with pytest.raises(InferenceSchemaError, match="title"):
            validate_output(TASK_COMPILE, _j(data))

    def test_missing_content_raises(self):
        data = self._valid()
        del data["content"]
        with pytest.raises(InferenceSchemaError, match="content"):
            validate_output(TASK_COMPILE, _j(data))

    def test_missing_source_relationships_raises(self):
        data = self._valid()
        del data["source_relationships"]
        with pytest.raises(InferenceSchemaError, match="source_relationships"):
            validate_output(TASK_COMPILE, _j(data))

    def test_invalid_relationship_in_list_raises(self):
        data = self._valid(source_relationships=[{"source_id": "x", "relationship": "INVALID"}])
        with pytest.raises(InferenceSchemaError, match="INVALID"):
            validate_output(TASK_COMPILE, _j(data))

    def test_all_valid_relationship_values(self):
        for rel in RELATIONSHIP_ENUM:
            data = self._valid(source_relationships=[{"source_id": "x", "relationship": rel}])
            result = validate_output(TASK_COMPILE, _j(data))
            assert result["source_relationships"][0]["relationship"] == rel

    def test_empty_source_relationships_ok(self):
        data = self._valid(source_relationships=[])
        result = validate_output(TASK_COMPILE, _j(data))
        assert result["source_relationships"] == []

    def test_extra_fields_ignored(self):
        data = self._valid()
        data["unexpected_field"] = "ignored"
        result = validate_output(TASK_COMPILE, _j(data))
        assert "unexpected_field" in result  # not stripped, just ignored (no error)

    def test_markdown_fence_stripped(self):
        data = self._valid()
        fenced = f"```json\n{json.dumps(data)}\n```"
        result = validate_output(TASK_COMPILE, fenced)
        assert result["title"] == "Test Article"

    def test_markdown_fence_without_lang_stripped(self):
        data = self._valid()
        fenced = f"```\n{json.dumps(data)}\n```"
        result = validate_output(TASK_COMPILE, fenced)
        assert result["content"] == "Some content."

    def test_source_relationships_not_list_raises(self):
        data = self._valid()
        data["source_relationships"] = "not a list"
        with pytest.raises(InferenceSchemaError):
            validate_output(TASK_COMPILE, _j(data))

    def test_invalid_json_raises(self):
        with pytest.raises(InferenceSchemaError, match="not valid JSON"):
            validate_output(TASK_COMPILE, "not json at all")

    def test_non_object_raises(self):
        with pytest.raises(InferenceSchemaError, match="JSON object"):
            validate_output(TASK_COMPILE, '["list", "not", "object"]')


# ---------------------------------------------------------------------------
# validate_output — TASK_UPDATE
# ---------------------------------------------------------------------------


class TestValidateUpdate:
    def _valid(self) -> dict:
        return {
            "content": "Updated content.",
            "relationship": "confirms",
            "changes_summary": "Added new information.",
        }

    def test_valid_output_returns_dict(self):
        result = validate_output(TASK_UPDATE, _j(self._valid()))
        assert result["content"] == "Updated content."
        assert result["relationship"] == "confirms"
        assert result["changes_summary"] == "Added new information."

    def test_missing_content_raises(self):
        data = self._valid()
        del data["content"]
        with pytest.raises(InferenceSchemaError, match="content"):
            validate_output(TASK_UPDATE, _j(data))

    def test_missing_relationship_raises(self):
        data = self._valid()
        del data["relationship"]
        with pytest.raises(InferenceSchemaError, match="relationship"):
            validate_output(TASK_UPDATE, _j(data))

    def test_missing_changes_summary_raises(self):
        data = self._valid()
        del data["changes_summary"]
        with pytest.raises(InferenceSchemaError, match="changes_summary"):
            validate_output(TASK_UPDATE, _j(data))

    def test_invalid_relationship_raises(self):
        data = self._valid()
        data["relationship"] = "invented_value"
        with pytest.raises(InferenceSchemaError, match="invented_value"):
            validate_output(TASK_UPDATE, _j(data))

    def test_all_valid_relationships(self):
        for rel in RELATIONSHIP_ENUM:
            data = self._valid()
            data["relationship"] = rel
            result = validate_output(TASK_UPDATE, _j(data))
            assert result["relationship"] == rel

    def test_extra_fields_ignored(self):
        data = self._valid()
        data["extra"] = "value"
        result = validate_output(TASK_UPDATE, _j(data))
        assert result["content"] == "Updated content."

    def test_markdown_fence_stripped(self):
        data = self._valid()
        fenced = f"```json\n{json.dumps(data)}\n```"
        result = validate_output(TASK_UPDATE, fenced)
        assert result["relationship"] == "confirms"


# ---------------------------------------------------------------------------
# validate_output — TASK_CLASSIFY
# ---------------------------------------------------------------------------


class TestValidateClassify:
    def _valid(self) -> dict:
        return {
            "relationship": "confirms",
            "confidence": 0.85,
            "reasoning": "Source corroborates existing claims.",
        }

    def test_valid_output_returns_dict(self):
        result = validate_output(TASK_CLASSIFY, _j(self._valid()))
        assert result["relationship"] == "confirms"
        assert result["confidence"] == 0.85
        assert result["reasoning"] == "Source corroborates existing claims."

    def test_missing_relationship_raises(self):
        data = self._valid()
        del data["relationship"]
        with pytest.raises(InferenceSchemaError, match="relationship"):
            validate_output(TASK_CLASSIFY, _j(data))

    def test_missing_confidence_raises(self):
        data = self._valid()
        del data["confidence"]
        with pytest.raises(InferenceSchemaError, match="confidence"):
            validate_output(TASK_CLASSIFY, _j(data))

    def test_missing_reasoning_raises(self):
        data = self._valid()
        del data["reasoning"]
        with pytest.raises(InferenceSchemaError, match="reasoning"):
            validate_output(TASK_CLASSIFY, _j(data))

    def test_invalid_relationship_raises(self):
        data = self._valid()
        data["relationship"] = "unknown_rel"
        with pytest.raises(InferenceSchemaError, match="unknown_rel"):
            validate_output(TASK_CLASSIFY, _j(data))

    def test_all_valid_relationships(self):
        for rel in RELATIONSHIP_ENUM:
            data = self._valid()
            data["relationship"] = rel
            result = validate_output(TASK_CLASSIFY, _j(data))
            assert result["relationship"] == rel

    def test_extra_fields_are_not_error(self):
        data = self._valid()
        data["bonus"] = 42
        result = validate_output(TASK_CLASSIFY, _j(data))
        assert result["confidence"] == 0.85

    def test_markdown_fence_stripped(self):
        data = self._valid()
        fenced = f"```json\n{json.dumps(data)}\n```"
        result = validate_output(TASK_CLASSIFY, fenced)
        assert result["relationship"] == "confirms"


# ---------------------------------------------------------------------------
# validate_output — TASK_CONTENTION
# ---------------------------------------------------------------------------


class TestValidateContention:
    def _valid(self) -> dict:
        return {
            "is_contention": True,
            "materiality": 0.7,
            "explanation": "Source directly contradicts the article.",
        }

    def test_valid_output_returns_dict(self):
        result = validate_output(TASK_CONTENTION, _j(self._valid()))
        assert result["is_contention"] is True
        assert result["materiality"] == 0.7
        assert result["explanation"] == "Source directly contradicts the article."

    def test_missing_is_contention_raises(self):
        data = self._valid()
        del data["is_contention"]
        with pytest.raises(InferenceSchemaError, match="is_contention"):
            validate_output(TASK_CONTENTION, _j(data))

    def test_missing_materiality_raises(self):
        data = self._valid()
        del data["materiality"]
        with pytest.raises(InferenceSchemaError, match="materiality"):
            validate_output(TASK_CONTENTION, _j(data))

    def test_missing_explanation_raises(self):
        data = self._valid()
        del data["explanation"]
        with pytest.raises(InferenceSchemaError, match="explanation"):
            validate_output(TASK_CONTENTION, _j(data))

    def test_false_contention(self):
        data = self._valid()
        data["is_contention"] = False
        data["materiality"] = 0.0
        data["explanation"] = None
        result = validate_output(TASK_CONTENTION, _j(data))
        assert result["is_contention"] is False

    def test_string_true_coerced_to_bool(self):
        data = self._valid()
        data["is_contention"] = "true"
        result = validate_output(TASK_CONTENTION, _j(data))
        assert result["is_contention"] is True

    def test_string_false_coerced_to_bool(self):
        data = self._valid()
        data["is_contention"] = "false"
        result = validate_output(TASK_CONTENTION, _j(data))
        assert result["is_contention"] is False

    def test_integer_is_contention_raises(self):
        data = self._valid()
        data["is_contention"] = 1  # not bool or string
        with pytest.raises(InferenceSchemaError, match="is_contention"):
            validate_output(TASK_CONTENTION, _j(data))

    def test_extra_fields_are_not_error(self):
        data = self._valid()
        data["contention_type"] = "contradiction"
        result = validate_output(TASK_CONTENTION, _j(data))
        assert result["materiality"] == 0.7

    def test_markdown_fence_stripped(self):
        data = self._valid()
        fenced = f"```json\n{json.dumps(data)}\n```"
        result = validate_output(TASK_CONTENTION, fenced)
        assert result["is_contention"] is True


# ---------------------------------------------------------------------------
# validate_output — TASK_SPLIT
# ---------------------------------------------------------------------------


class TestValidateSplit:
    def _valid(self) -> dict:
        return {
            "split_index": 512,
            "part_a_title": "Part One",
            "part_b_title": "Part Two",
            "reasoning": "Natural topic boundary at paragraph 3.",
        }

    def test_valid_output_returns_dict(self):
        result = validate_output(TASK_SPLIT, _j(self._valid()))
        assert result["split_index"] == 512
        assert result["part_a_title"] == "Part One"
        assert result["part_b_title"] == "Part Two"
        assert result["reasoning"] == "Natural topic boundary at paragraph 3."

    def test_missing_split_index_raises(self):
        data = self._valid()
        del data["split_index"]
        with pytest.raises(InferenceSchemaError, match="split_index"):
            validate_output(TASK_SPLIT, _j(data))

    def test_missing_part_a_title_raises(self):
        data = self._valid()
        del data["part_a_title"]
        with pytest.raises(InferenceSchemaError, match="part_a_title"):
            validate_output(TASK_SPLIT, _j(data))

    def test_missing_part_b_title_raises(self):
        data = self._valid()
        del data["part_b_title"]
        with pytest.raises(InferenceSchemaError, match="part_b_title"):
            validate_output(TASK_SPLIT, _j(data))

    def test_missing_reasoning_raises(self):
        data = self._valid()
        del data["reasoning"]
        with pytest.raises(InferenceSchemaError, match="reasoning"):
            validate_output(TASK_SPLIT, _j(data))

    def test_split_index_not_int_raises(self):
        data = self._valid()
        data["split_index"] = "not-an-int"
        with pytest.raises(InferenceSchemaError, match="split_index"):
            validate_output(TASK_SPLIT, _j(data))

    def test_split_index_zero_ok(self):
        data = self._valid()
        data["split_index"] = 0
        result = validate_output(TASK_SPLIT, _j(data))
        assert result["split_index"] == 0

    def test_extra_fields_are_not_error(self):
        data = self._valid()
        data["extra"] = "value"
        result = validate_output(TASK_SPLIT, _j(data))
        assert result["split_index"] == 512

    def test_markdown_fence_stripped(self):
        data = self._valid()
        fenced = f"```json\n{json.dumps(data)}\n```"
        result = validate_output(TASK_SPLIT, fenced)
        assert result["split_index"] == 512


# ---------------------------------------------------------------------------
# validate_output — common edge cases
# ---------------------------------------------------------------------------


class TestValidateCommon:
    def test_unknown_task_type_raises(self):
        with pytest.raises(InferenceSchemaError, match="Unknown task type"):
            validate_output("not_a_task", '{"key": "val"}')

    def test_bare_string_raises(self):
        with pytest.raises(InferenceSchemaError):
            validate_output(TASK_COMPILE, '"just a string"')

    def test_number_raises(self):
        with pytest.raises(InferenceSchemaError):
            validate_output(TASK_COMPILE, "42")

    def test_empty_string_raises(self):
        with pytest.raises(InferenceSchemaError):
            validate_output(TASK_COMPILE, "")

    def test_whitespace_only_raises(self):
        with pytest.raises(InferenceSchemaError):
            validate_output(TASK_COMPILE, "   ")

    def test_fence_then_non_json_raises(self):
        with pytest.raises(InferenceSchemaError):
            validate_output(TASK_COMPILE, "```json\nnot valid json\n```")


# ---------------------------------------------------------------------------
# InferenceProvider.infer() — schema validation integration
# ---------------------------------------------------------------------------


class TestInferenceProviderSchemaValidation:
    async def test_valid_compile_response_populates_parsed(self):
        p = InferenceProvider()
        response = json.dumps({"title": "T", "content": "C", "source_relationships": [{"source_id": "x", "relationship": "originates"}]})
        p.configure(lambda prompt: response)
        result = await p.infer(TASK_COMPILE, "prompt")
        assert not result.degraded
        assert result.parsed is not None
        assert result.parsed["title"] == "T"

    async def test_invalid_compile_response_returns_content_but_no_parsed(self):
        p = InferenceProvider()
        # Missing required fields
        bad_response = json.dumps({"only_key": "present"})
        p.configure(lambda prompt: bad_response)
        result = await p.infer(TASK_COMPILE, "prompt")
        assert not result.degraded  # not degraded — backend worked
        assert result.content == bad_response
        assert result.parsed is None  # schema validation failed

    async def test_valid_update_response_populates_parsed(self):
        p = InferenceProvider()
        response = json.dumps({"content": "Updated.", "relationship": "confirms", "changes_summary": "Minor fix."})
        p.configure(lambda prompt: response)
        result = await p.infer(TASK_UPDATE, "prompt")
        assert result.parsed is not None
        assert result.parsed["relationship"] == "confirms"

    async def test_valid_contention_response_populates_parsed(self):
        p = InferenceProvider()
        response = json.dumps({"is_contention": True, "materiality": 0.6, "explanation": "Disagrees."})
        p.configure(lambda prompt: response)
        result = await p.infer(TASK_CONTENTION, "prompt")
        assert result.parsed is not None
        assert result.parsed["is_contention"] is True

    async def test_valid_split_response_populates_parsed(self):
        p = InferenceProvider()
        response = json.dumps({"split_index": 100, "part_a_title": "A", "part_b_title": "B", "reasoning": "R"})
        p.configure(lambda prompt: response)
        result = await p.infer(TASK_SPLIT, "prompt")
        assert result.parsed is not None
        assert result.parsed["split_index"] == 100

    async def test_valid_classify_response_populates_parsed(self):
        p = InferenceProvider()
        response = json.dumps({"relationship": "supersedes", "confidence": 0.9, "reasoning": "Newer data."})
        p.configure(lambda prompt: response)
        result = await p.infer(TASK_CLASSIFY, "prompt")
        assert result.parsed is not None
        assert result.parsed["relationship"] == "supersedes"

    async def test_degraded_result_has_no_parsed(self):
        p = InferenceProvider()  # no backend
        result = await p.infer(TASK_COMPILE, "prompt")
        assert result.degraded
        assert result.parsed is None

    async def test_backend_exception_sets_degraded_no_parsed(self):
        p = InferenceProvider()

        def boom(prompt):
            raise RuntimeError("boom")

        p.configure(boom)
        result = await p.infer(TASK_COMPILE, "prompt")
        assert result.degraded
        assert result.parsed is None
