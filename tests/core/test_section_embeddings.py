# SPDX-License-Identifier: MIT
"""Tests for valence.core.section_embeddings."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from valence.core.section_embeddings import _flatten_tree


class TestFlattenTree:
    def test_single_node(self):
        tree = [{"title": "Root", "summary": "s", "start_char": 0, "end_char": 100, "children": []}]
        flat = _flatten_tree(tree)
        assert len(flat) == 1
        assert flat[0]["tree_path"] == "0"
        assert flat[0]["depth"] == 0
        assert flat[0]["start_char"] == 0
        assert flat[0]["end_char"] == 100

    def test_nested_tree(self):
        tree = [
            {
                "title": "A",
                "summary": "",
                "start_char": 0,
                "end_char": 200,
                "children": [
                    {"title": "A.0", "summary": "", "start_char": 0, "end_char": 100, "children": []},
                    {"title": "A.1", "summary": "", "start_char": 100, "end_char": 200, "children": []},
                ],
            },
            {
                "title": "B",
                "summary": "",
                "start_char": 200,
                "end_char": 300,
                "children": [],
            },
        ]
        flat = _flatten_tree(tree)
        assert len(flat) == 4
        paths = [n["tree_path"] for n in flat]
        assert paths == ["0", "0.0", "0.1", "1"]

    def test_deep_nesting(self):
        tree = [
            {
                "title": "Root",
                "summary": "",
                "start_char": 0,
                "end_char": 300,
                "children": [
                    {
                        "title": "Child",
                        "summary": "",
                        "start_char": 0,
                        "end_char": 150,
                        "children": [
                            {
                                "title": "Grandchild",
                                "summary": "",
                                "start_char": 0,
                                "end_char": 75,
                                "children": [],
                            }
                        ],
                    }
                ],
            }
        ]
        flat = _flatten_tree(tree)
        assert len(flat) == 3
        assert flat[0]["tree_path"] == "0"
        assert flat[0]["depth"] == 0
        assert flat[1]["tree_path"] == "0.0"
        assert flat[1]["depth"] == 1
        assert flat[2]["tree_path"] == "0.0.0"
        assert flat[2]["depth"] == 2

    def test_multiple_roots(self):
        tree = [
            {"title": "A", "summary": "", "start_char": 0, "end_char": 100, "children": []},
            {"title": "B", "summary": "", "start_char": 100, "end_char": 200, "children": []},
            {"title": "C", "summary": "", "start_char": 200, "end_char": 300, "children": []},
        ]
        flat = _flatten_tree(tree)
        assert len(flat) == 3
        assert [n["tree_path"] for n in flat] == ["0", "1", "2"]

    def test_empty_tree(self):
        assert _flatten_tree([]) == []


class TestEmbedSourceSections:
    """Test embed_source_sections with mocked DB and embeddings."""

    @pytest.fixture
    def mock_db(self):
        """Set up mock cursor that returns a source row."""
        mock_cm = MagicMock()
        mock_cur = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_cur)
        mock_cm.__exit__ = MagicMock(return_value=False)
        return mock_cm, mock_cur

    @patch("valence.core.section_embeddings.generate_embedding")
    @patch("valence.core.section_embeddings.get_cursor")
    @pytest.mark.asyncio
    async def test_embed_with_existing_tree(self, mock_gc, mock_embed):
        """Source with tree_index in metadata gets sections embedded."""
        mock_cm = MagicMock()
        mock_cur = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_cur)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_gc.return_value = mock_cm

        # First call: fetch source
        source_row = {
            "id": "src-1",
            "content": "Hello world this is a test document with enough content.",
            "metadata": {
                "tree_index": [
                    {
                        "title": "Greeting",
                        "summary": "A greeting",
                        "start_char": 0,
                        "end_char": 11,
                        "children": [],
                    },
                    {
                        "title": "Content",
                        "summary": "Main content",
                        "start_char": 12,
                        "end_char": 56,
                        "children": [],
                    },
                ]
            },
        }
        # mock_cur.fetchone returns source row first, then None for section lookups
        mock_cur.fetchone = MagicMock(side_effect=[source_row, None, None])
        mock_embed.return_value = [0.1] * 1536

        from valence.core.section_embeddings import embed_source_sections

        result = await embed_source_sections("src-1")
        assert result.success
        assert result.data["sections_embedded"] == 2
        assert mock_embed.call_count == 2

    @patch("valence.core.section_embeddings.get_cursor")
    @pytest.mark.asyncio
    async def test_source_not_found(self, mock_gc):
        mock_cm = MagicMock()
        mock_cur = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_cur)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_gc.return_value = mock_cm
        mock_cur.fetchone.return_value = None

        from valence.core.section_embeddings import embed_source_sections

        result = await embed_source_sections("nonexistent")
        assert not result.success
        assert "not found" in result.error.lower() or "Failed" in result.error

    @patch("valence.core.section_embeddings.get_cursor")
    @pytest.mark.asyncio
    async def test_empty_content_skipped(self, mock_gc):
        mock_cm = MagicMock()
        mock_cur = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_cur)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_gc.return_value = mock_cm
        mock_cur.fetchone.return_value = {
            "id": "src-1",
            "content": "   ",
            "metadata": {},
        }

        from valence.core.section_embeddings import embed_source_sections

        result = await embed_source_sections("src-1")
        assert result.success
        assert result.data["sections_embedded"] == 0
