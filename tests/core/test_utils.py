"""Tests for core utilities."""

import pytest

from valence.core.utils import escape_ilike


class TestEscapeIlike:
    """Tests for escape_ilike function."""

    def test_plain_text_unchanged(self):
        """Plain text without metacharacters passes through unchanged."""
        assert escape_ilike("hello world") == "hello world"
        assert escape_ilike("simple query") == "simple query"
        assert escape_ilike("") == ""

    def test_escapes_percent(self):
        """Percent signs are escaped."""
        assert escape_ilike("100%") == "100\\%"
        assert escape_ilike("%match%") == "\\%match\\%"
        assert escape_ilike("50% off") == "50\\% off"

    def test_escapes_underscore(self):
        """Underscores are escaped."""
        assert escape_ilike("snake_case") == "snake\\_case"
        assert escape_ilike("_prefix") == "\\_prefix"
        assert escape_ilike("suffix_") == "suffix\\_"
        assert escape_ilike("a_b_c") == "a\\_b\\_c"

    def test_escapes_backslash(self):
        """Backslashes are escaped."""
        assert escape_ilike("path\\to\\file") == "path\\\\to\\\\file"
        assert escape_ilike("\\start") == "\\\\start"
        assert escape_ilike("end\\") == "end\\\\"

    def test_escapes_combined_metacharacters(self):
        """Multiple metacharacters are all escaped."""
        assert escape_ilike("100% of _all_") == "100\\% of \\_all\\_"
        assert escape_ilike("%_\\") == "\\%\\_\\\\"
        assert escape_ilike("a_b%c\\d") == "a\\_b\\%c\\\\d"

    def test_backslash_escaped_before_others(self):
        """Backslash must be escaped first to avoid double-escaping.

        If we escaped % before \\, then "\\%" would become "\\\\%" (wrong).
        Correct order: \\ -> \\\\ first, then % -> \\%.
        """
        # Input: \%  (backslash followed by percent)
        # Correct: \\% -> \\\\\\% (escaped backslash + escaped percent)
        assert escape_ilike("\\%") == "\\\\\\%"
        assert escape_ilike("\\_") == "\\\\\\_"

    def test_preserves_safe_special_chars(self):
        """Characters that aren't ILIKE metacharacters are preserved."""
        assert escape_ilike("user@email.com") == "user@email.com"
        assert escape_ilike("hello! (test)") == "hello! (test)"
        assert escape_ilike("[brackets]") == "[brackets]"
        assert escape_ilike("a*b+c?") == "a*b+c?"

    def test_unicode_preserved(self):
        """Unicode characters pass through unchanged."""
        assert escape_ilike("日本語") == "日本語"
        assert escape_ilike("émoji 🎉") == "émoji 🎉"
        assert escape_ilike("100% 日本語") == "100\\% 日本語"
