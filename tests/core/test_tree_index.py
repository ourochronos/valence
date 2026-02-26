# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Tests for tree_index module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from valence.core.tree_index import (
    _estimate_tokens,
    _validate_tree,
    build_tree_index,
    get_tree_index,
    get_tree_region,
)


class TestEstimateTokens:
    def test_basic(self):
        text = "a" * 350
        assert _estimate_tokens(text) == 100

    def test_empty(self):
        assert _estimate_tokens("") == 0


class TestValidateTree:
    def test_valid_tree(self):
        tree = {
            "nodes": [
                {"title": "A", "start_char": 0, "end_char": 50},
                {"title": "B", "start_char": 50, "end_char": 100},
            ]
        }
        issues = _validate_tree(tree, 100)
        assert issues == []

    def test_clamps_overshoot(self):
        tree = {
            "nodes": [
                {"title": "A", "start_char": 0, "end_char": 150},
            ]
        }
        issues = _validate_tree(tree, 100)
        assert issues == []
        assert tree["nodes"][0]["end_char"] == 100

    def test_invalid_range(self):
        tree = {
            "nodes": [
                {"title": "Bad", "start_char": 50, "end_char": 50},
            ]
        }
        issues = _validate_tree(tree, 100)
        assert len(issues) == 1
        assert "Invalid range" in issues[0]

    def test_missing_offsets(self):
        tree = {"nodes": [{"title": "No offsets"}]}
        issues = _validate_tree(tree, 100)
        assert any("Missing" in i for i in issues)

    def test_nested_validation(self):
        tree = {
            "nodes": [
                {
                    "title": "Parent",
                    "start_char": 0,
                    "end_char": 100,
                    "children": [
                        {"title": "Child", "start_char": 0, "end_char": 50},
                    ],
                },
            ]
        }
        issues = _validate_tree(tree, 100)
        assert issues == []


def _make_cursor(fetchone_seq=None):
    cur = MagicMock()
    cur.__enter__ = MagicMock(return_value=cur)
    cur.__exit__ = MagicMock(return_value=False)
    if fetchone_seq is not None:
        cur.fetchone = MagicMock(side_effect=fetchone_seq)
    return cur


class TestBuildTreeIndex:
    @pytest.mark.asyncio
    async def test_source_not_found(self):
        cur = _make_cursor(fetchone_seq=[None])
        with patch("valence.core.tree_index.get_cursor", return_value=cur):
            result = await build_tree_index("00000000-0000-0000-0000-000000000000")
        assert not result.success
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_empty_content(self):
        cur = _make_cursor(fetchone_seq=[{"id": "abc", "content": "", "metadata": {}}])
        with patch("valence.core.tree_index.get_cursor", return_value=cur):
            result = await build_tree_index("abc")
        assert not result.success
        assert "no content" in result.error

    @pytest.mark.asyncio
    async def test_already_indexed(self):
        cur = _make_cursor(fetchone_seq=[{"id": "abc", "content": "text", "metadata": {"tree_index": {"nodes": []}}}])
        with patch("valence.core.tree_index.get_cursor", return_value=cur):
            result = await build_tree_index("abc")
        assert not result.success
        assert "already has tree_index" in result.error

    @pytest.mark.asyncio
    async def test_single_window_success(self):
        source_content = "Hello world " * 100
        tree_response = json.dumps({"nodes": [{"title": "Test", "summary": "A test", "start_char": 0, "end_char": len(source_content)}]})

        cur = _make_cursor(
            fetchone_seq=[
                {"id": "abc", "content": source_content, "metadata": {}},
                {"id": "abc"},  # UPDATE RETURNING
                None,  # second UPDATE
            ]
        )
        with (
            patch("valence.core.tree_index.get_cursor", return_value=cur),
            patch("valence.core.tree_index._call_llm", return_value=tree_response),
        ):
            result = await build_tree_index("abc")

        assert result.success
        assert result.data["method"] == "single"
        assert result.data["node_count"] == 1


class TestGetTreeIndex:
    @pytest.mark.asyncio
    async def test_not_found(self):
        cur = _make_cursor(fetchone_seq=[None])
        with patch("valence.core.tree_index.get_cursor", return_value=cur):
            result = await get_tree_index("abc")
        assert not result.success

    @pytest.mark.asyncio
    async def test_no_index(self):
        cur = _make_cursor(fetchone_seq=[{"metadata": {}}])
        with patch("valence.core.tree_index.get_cursor", return_value=cur):
            result = await get_tree_index("abc")
        assert not result.success
        assert "no tree index" in result.error

    @pytest.mark.asyncio
    async def test_returns_tree(self):
        tree = {"nodes": [{"title": "A", "start_char": 0, "end_char": 100}]}
        cur = _make_cursor(fetchone_seq=[{"metadata": {"tree_index": tree}}])
        with patch("valence.core.tree_index.get_cursor", return_value=cur):
            result = await get_tree_index("abc")
        assert result.success
        assert result.data == tree


class TestGetTreeRegion:
    @pytest.mark.asyncio
    async def test_not_found(self):
        cur = _make_cursor(fetchone_seq=[None])
        with patch("valence.core.tree_index.get_cursor", return_value=cur):
            result = await get_tree_region("abc", 0, 10)
        assert not result.success

    @pytest.mark.asyncio
    async def test_invalid_range(self):
        cur = _make_cursor(fetchone_seq=[{"content": "short"}])
        with patch("valence.core.tree_index.get_cursor", return_value=cur):
            result = await get_tree_region("abc", 0, 1000)
        assert not result.success
        assert "Invalid range" in result.error

    @pytest.mark.asyncio
    async def test_extracts_region(self):
        content = "Hello world, this is a test document with some content."
        cur = _make_cursor(fetchone_seq=[{"content": content}])
        with patch("valence.core.tree_index.get_cursor", return_value=cur):
            result = await get_tree_region("abc", 6, 11)
        assert result.success
        assert result.data["text"] == "world"
