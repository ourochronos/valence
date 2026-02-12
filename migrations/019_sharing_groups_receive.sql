-- Migration 019: Sharing groups, received shares, and notifications (#342, #344)

-- MLS-backed sharing groups for work_with_me intent
CREATE TABLE IF NOT EXISTS sharing_groups (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    creator_did TEXT NOT NULL,
    mls_group_id TEXT NOT NULL UNIQUE,
    intent TEXT NOT NULL DEFAULT 'work_with_me',
    epoch INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT sharing_groups_valid_intent
        CHECK (intent IN ('work_with_me'))
);

CREATE INDEX IF NOT EXISTS idx_sharing_groups_creator ON sharing_groups(creator_did);

-- Group membership
CREATE TABLE IF NOT EXISTS sharing_group_members (
    group_id UUID NOT NULL REFERENCES sharing_groups(id) ON DELETE CASCADE,
    member_did TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'member',
    joined_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (group_id, member_did),

    CONSTRAINT sharing_group_members_valid_role
        CHECK (role IN ('creator', 'admin', 'member'))
);

CREATE INDEX IF NOT EXISTS idx_sharing_group_members_did ON sharing_group_members(member_did);

-- Received shares from federation peers
CREATE TABLE IF NOT EXISTS received_shares (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    belief_id UUID NOT NULL,
    sender_did TEXT NOT NULL,
    recipient_did TEXT NOT NULL,
    content TEXT NOT NULL,
    confidence JSONB,
    intent TEXT NOT NULL,
    consent_chain JSONB NOT NULL,
    encrypted_envelope JSONB,
    status TEXT NOT NULL DEFAULT 'pending',
    received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT received_shares_valid_status
        CHECK (status IN ('pending', 'accepted', 'rejected')),
    CONSTRAINT received_shares_valid_intent
        CHECK (intent IN ('know_me', 'work_with_me', 'learn_from_me', 'use_this'))
);

CREATE INDEX IF NOT EXISTS idx_received_shares_recipient ON received_shares(recipient_did);
CREATE INDEX IF NOT EXISTS idx_received_shares_sender ON received_shares(sender_did);
CREATE INDEX IF NOT EXISTS idx_received_shares_belief ON received_shares(belief_id);

-- Notifications for incoming shares
CREATE TABLE IF NOT EXISTS share_notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    share_id UUID NOT NULL REFERENCES received_shares(id) ON DELETE CASCADE,
    recipient_did TEXT NOT NULL,
    sender_did TEXT NOT NULL,
    belief_id UUID NOT NULL,
    intent TEXT NOT NULL,
    read BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_share_notifications_recipient ON share_notifications(recipient_did);
CREATE INDEX IF NOT EXISTS idx_share_notifications_unread ON share_notifications(recipient_did)
    WHERE read = FALSE;
