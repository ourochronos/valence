"""Tests for privacy migration - visibility to SharePolicy conversion."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

from valence.privacy.migration import (
    migrate_visibility,
    get_share_policy_json,
    migrate_all_beliefs,
    migrate_all_beliefs_sync,
)
from valence.privacy.types import ShareLevel, EnforcementType


class TestMigrateVisibility:
    """Tests for migrate_visibility function."""
    
    def test_private_lowercase(self):
        """Test private visibility migration."""
        result = migrate_visibility("private")
        
        assert result["level"] == "private"
        assert result["enforcement"] == "cryptographic"
        assert result["recipients"] is None
        assert result["propagation"] is None
    
    def test_private_uppercase(self):
        """Test PRIVATE visibility migration."""
        result = migrate_visibility("PRIVATE")
        
        assert result["level"] == "private"
        assert result["enforcement"] == "cryptographic"
    
    def test_federated_lowercase(self):
        """Test federated visibility migration."""
        result = migrate_visibility("federated")
        
        assert result["level"] == "bounded"
        assert result["enforcement"] == "cryptographic"
        assert result["propagation"] is not None
        assert result["propagation"]["allowed_domains"] == ["federation"]
    
    def test_federated_uppercase(self):
        """Test FEDERATED visibility migration."""
        result = migrate_visibility("FEDERATED")
        
        assert result["level"] == "bounded"
        assert result["enforcement"] == "cryptographic"
        assert result["propagation"]["allowed_domains"] == ["federation"]
    
    def test_public_lowercase(self):
        """Test public visibility migration."""
        result = migrate_visibility("public")
        
        assert result["level"] == "public"
        assert result["enforcement"] == "honor"
        assert result["recipients"] is None
        assert result["propagation"] is None
    
    def test_public_uppercase(self):
        """Test PUBLIC visibility migration."""
        result = migrate_visibility("PUBLIC")
        
        assert result["level"] == "public"
        assert result["enforcement"] == "honor"
    
    def test_unknown_defaults_to_private(self):
        """Test that unknown visibility defaults to private."""
        result = migrate_visibility("unknown_value")
        
        assert result["level"] == "private"
        assert result["enforcement"] == "cryptographic"
    
    def test_empty_string_defaults_to_private(self):
        """Test that empty string defaults to private."""
        result = migrate_visibility("")
        
        assert result["level"] == "private"
        assert result["enforcement"] == "cryptographic"
    
    def test_none_defaults_to_private(self):
        """Test that None defaults to private."""
        result = migrate_visibility(None)
        
        assert result["level"] == "private"
        assert result["enforcement"] == "cryptographic"
    
    def test_result_is_json_serializable(self):
        """Test that all results are JSON serializable."""
        for visibility in ["private", "PRIVATE", "federated", "FEDERATED", 
                          "public", "PUBLIC", "unknown"]:
            result = migrate_visibility(visibility)
            # Should not raise
            json_str = json.dumps(result)
            assert json_str is not None
    
    def test_private_roundtrip(self):
        """Test that private migration can be deserialized back to SharePolicy."""
        from valence.privacy.types import SharePolicy
        
        result = migrate_visibility("private")
        policy = SharePolicy.from_dict(result)
        
        assert policy.level == ShareLevel.PRIVATE
        assert policy.enforcement == EnforcementType.CRYPTOGRAPHIC
    
    def test_federated_roundtrip(self):
        """Test that federated migration can be deserialized back to SharePolicy."""
        from valence.privacy.types import SharePolicy
        
        result = migrate_visibility("federated")
        policy = SharePolicy.from_dict(result)
        
        assert policy.level == ShareLevel.BOUNDED
        assert policy.enforcement == EnforcementType.CRYPTOGRAPHIC
        assert policy.propagation.allowed_domains == ["federation"]
    
    def test_public_roundtrip(self):
        """Test that public migration can be deserialized back to SharePolicy."""
        from valence.privacy.types import SharePolicy
        
        result = migrate_visibility("public")
        policy = SharePolicy.from_dict(result)
        
        assert policy.level == ShareLevel.PUBLIC
        assert policy.enforcement == EnforcementType.HONOR


class TestGetSharePolicyJson:
    """Tests for get_share_policy_json function."""
    
    def test_private_json(self):
        """Test JSON output for private visibility."""
        result = get_share_policy_json("private")
        parsed = json.loads(result)
        
        assert parsed["level"] == "private"
        assert parsed["enforcement"] == "cryptographic"
    
    def test_federated_json(self):
        """Test JSON output for federated visibility."""
        result = get_share_policy_json("federated")
        parsed = json.loads(result)
        
        assert parsed["level"] == "bounded"
        assert parsed["enforcement"] == "cryptographic"
        assert parsed["propagation"]["allowed_domains"] == ["federation"]
    
    def test_public_json(self):
        """Test JSON output for public visibility."""
        result = get_share_policy_json("public")
        parsed = json.loads(result)
        
        assert parsed["level"] == "public"
        assert parsed["enforcement"] == "honor"
    
    def test_unknown_defaults_to_private(self):
        """Test that unknown visibility defaults to private JSON."""
        result = get_share_policy_json("invalid")
        parsed = json.loads(result)
        
        assert parsed["level"] == "private"
        assert parsed["enforcement"] == "cryptographic"
    
    def test_all_results_valid_json(self):
        """Test that all visibility values produce valid JSON."""
        for visibility in ["private", "PRIVATE", "federated", "FEDERATED",
                          "public", "PUBLIC", "unknown", "", None]:
            result = get_share_policy_json(visibility)
            # Should not raise
            parsed = json.loads(result)
            assert "level" in parsed
            assert "enforcement" in parsed


class TestMigrateAllBeliefs:
    """Tests for migrate_all_beliefs async function."""
    
    @pytest.mark.asyncio
    async def test_no_beliefs_need_migration(self):
        """Test when all beliefs already have share_policy."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(side_effect=[100, 0])  # total=100, needs_migration=0
        
        result = await migrate_all_beliefs(mock_conn)
        
        assert result["total"] == 100
        assert result["needed_migration"] == 0
        assert result["migrated"] == 0
        # Should not call execute for migration
        mock_conn.execute.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_migration_runs_update(self):
        """Test that migration runs UPDATE when beliefs need migration."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(side_effect=[100, 50, 100])  # total, needs, migrated
        
        result = await migrate_all_beliefs(mock_conn)
        
        assert result["total"] == 100
        assert result["needed_migration"] == 50
        assert result["migrated"] == 100
        # Should call execute with UPDATE statement
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0][0]
        assert "UPDATE beliefs" in call_args
        assert "SET share_policy" in call_args
    
    @pytest.mark.asyncio
    async def test_migration_uses_case_statement(self):
        """Test that migration uses CASE statement for all visibility values."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(side_effect=[10, 10, 10])
        
        await migrate_all_beliefs(mock_conn)
        
        call_args = mock_conn.execute.call_args[0][0]
        # Check all visibility values are handled
        assert "WHEN 'private'" in call_args
        assert "WHEN 'PRIVATE'" in call_args
        assert "WHEN 'federated'" in call_args
        assert "WHEN 'FEDERATED'" in call_args
        assert "WHEN 'public'" in call_args
        assert "WHEN 'PUBLIC'" in call_args
        assert "ELSE" in call_args  # Default case


class TestMigrateAllBeliefsSync:
    """Tests for migrate_all_beliefs_sync function."""
    
    def test_no_beliefs_need_migration(self):
        """Test when all beliefs already have share_policy."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [(100,), (0,)]  # total, needs_migration
        
        result = migrate_all_beliefs_sync(mock_conn)
        
        assert result["total"] == 100
        assert result["needed_migration"] == 0
        assert result["migrated"] == 0
    
    def test_migration_commits(self):
        """Test that migration commits changes."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [(100,), (50,), (100,)]  # total, needs, migrated
        
        migrate_all_beliefs_sync(mock_conn)
        
        mock_conn.commit.assert_called_once()
    
    def test_cursor_closed_on_success(self):
        """Test that cursor is closed after successful migration."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [(100,), (0,)]
        
        migrate_all_beliefs_sync(mock_conn)
        
        mock_cursor.close.assert_called_once()
    
    def test_cursor_closed_on_error(self):
        """Test that cursor is closed even if error occurs."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = Exception("DB error")
        
        with pytest.raises(Exception):
            migrate_all_beliefs_sync(mock_conn)
        
        mock_cursor.close.assert_called_once()


class TestMigrationCorrectness:
    """Integration-style tests for migration correctness."""
    
    def test_all_visibility_levels_produce_valid_policies(self):
        """Test that all visibility levels produce valid SharePolicy objects."""
        from valence.privacy.types import SharePolicy
        
        visibility_levels = ["private", "PRIVATE", "federated", "FEDERATED", 
                           "public", "PUBLIC"]
        
        for visibility in visibility_levels:
            result = migrate_visibility(visibility)
            # Should not raise - validates the structure
            policy = SharePolicy.from_dict(result)
            
            # Verify the policy has expected attributes
            assert policy.level in ShareLevel
            assert policy.enforcement in EnforcementType
    
    def test_private_cannot_share(self):
        """Test that migrated private policy blocks sharing."""
        from valence.privacy.types import SharePolicy
        
        result = migrate_visibility("private")
        policy = SharePolicy.from_dict(result)
        
        assert not policy.allows_sharing_to("did:key:anyone")
    
    def test_public_can_share(self):
        """Test that migrated public policy allows sharing."""
        from valence.privacy.types import SharePolicy
        
        result = migrate_visibility("public")
        policy = SharePolicy.from_dict(result)
        
        assert policy.allows_sharing_to("did:key:anyone")
    
    def test_federated_has_bounded_scope(self):
        """Test that federated migration creates bounded scope with federation domain."""
        from valence.privacy.types import SharePolicy, ShareLevel
        
        result = migrate_visibility("federated")
        policy = SharePolicy.from_dict(result)
        
        assert policy.level == ShareLevel.BOUNDED
        assert policy.propagation is not None
        assert "federation" in policy.propagation.allowed_domains
