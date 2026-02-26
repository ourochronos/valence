"""Integration tests for Valence.

This package contains integration tests that verify the system works correctly
when components interact with real dependencies (database, HTTP, etc.).

Test Categories:
- test_deployment.py: Database schema and basic operations
- test_belief_lifecycle.py: Full belief CRUD and lifecycle
- test_federation_sync.py: Multi-node federation sync
- test_trust_propagation.py: Trust calculation and propagation

Running Integration Tests:
    # All integration tests (requires database)
    pytest tests/integration/ -m integration -v

    # Specific test file
    pytest tests/integration/test_belief_lifecycle.py -v

    # With Docker test environment
    make docker-up
    make test-integration
    make docker-down

Environment Variables:
    VALENCE_DB_HOST: Database host (default: localhost)
    VALENCE_DB_PORT: Database port (default: 5433)
    VALENCE_DB_NAME: Database name (default: valence_test)
    VALENCE_DB_USER: Database user (default: valence)
    VALENCE_DB_PASSWORD: Database password (default: testpass)
    VALENCE_PRIMARY_URL: Primary server URL for HTTP tests
    VALENCE_PEER_URL: Peer server URL for federation tests
"""
