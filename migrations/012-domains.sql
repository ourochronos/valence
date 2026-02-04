-- Domain schema for Valence
-- Domains are named scopes for trust (e.g., "research-team", "family")
-- Closes #62

-- Create domains table
CREATE TABLE IF NOT EXISTS domains (
    domain_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    owner_did TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Name must be unique per owner
    CONSTRAINT domains_name_owner_unique UNIQUE (name, owner_did)
);

-- Create domain_memberships table
CREATE TABLE IF NOT EXISTS domain_memberships (
    domain_id UUID NOT NULL REFERENCES domains(domain_id) ON DELETE CASCADE,
    member_did TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('owner', 'admin', 'member')),
    joined_at TIMESTAMPTZ DEFAULT NOW(),
    
    PRIMARY KEY (domain_id, member_did)
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_domains_owner ON domains(owner_did);
CREATE INDEX IF NOT EXISTS idx_domains_name ON domains(name);
CREATE INDEX IF NOT EXISTS idx_domain_memberships_member ON domain_memberships(member_did);
CREATE INDEX IF NOT EXISTS idx_domain_memberships_role ON domain_memberships(role);

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION update_domains_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS domains_updated ON domains;
CREATE TRIGGER domains_updated
    BEFORE UPDATE ON domains
    FOR EACH ROW
    EXECUTE FUNCTION update_domains_timestamp();

-- Comments for documentation
COMMENT ON TABLE domains IS 'Named scopes for trust - group members who share beliefs within a common context';
COMMENT ON COLUMN domains.name IS 'Domain name, unique per owner (e.g., "research-team", "family")';
COMMENT ON COLUMN domains.owner_did IS 'DID of the domain owner with full control';
COMMENT ON TABLE domain_memberships IS 'Maps DIDs to domains with their roles';
COMMENT ON COLUMN domain_memberships.role IS 'Member role: owner (full control), admin (manage members), member (basic access)';
