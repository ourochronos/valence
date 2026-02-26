"""Tests for error response helpers."""

import os
from unittest.mock import patch

from valence.server.errors import internal_error


class TestInternalError:
    def test_includes_request_id(self):
        resp = internal_error()
        data = resp.body.decode()
        import json

        body = json.loads(data)
        assert body["success"] is False
        assert "request_id" in body["error"]
        assert len(body["error"]["request_id"]) == 12

    def test_debug_mode_includes_exception_detail(self):
        with patch.dict(os.environ, {"VALENCE_DEBUG": "1"}):
            # Reimport to pick up env change
            import valence.server.errors as errors

            old = errors._DEBUG
            errors._DEBUG = True
            try:
                exc = ValueError("test value error")
                resp = errors.internal_error(exc=exc)
                import json

                body = json.loads(resp.body.decode())
                assert body["error"]["exception"] == "ValueError"
                assert body["error"]["detail"] == "test value error"
            finally:
                errors._DEBUG = old

    def test_production_mode_hides_exception_detail(self):
        import valence.server.errors as errors

        old = errors._DEBUG
        errors._DEBUG = False
        try:
            exc = ValueError("secret internal detail")
            resp = errors.internal_error(exc=exc)
            import json

            body = json.loads(resp.body.decode())
            assert "exception" not in body["error"]
            assert "detail" not in body["error"]
            assert "request_id" in body["error"]
        finally:
            errors._DEBUG = old

    def test_captures_current_exception(self):
        import valence.server.errors as errors

        old = errors._DEBUG
        errors._DEBUG = True
        try:
            try:
                raise TypeError("auto-captured")
            except TypeError:
                resp = errors.internal_error()
            import json

            body = json.loads(resp.body.decode())
            assert body["error"]["exception"] == "TypeError"
            assert body["error"]["detail"] == "auto-captured"
        finally:
            errors._DEBUG = old
