-- Migration 008: Fix mutation_queue FK blocking source pipeline batch mode
-- Issue #568: article_id FK prevents enqueuing source-pipeline tasks
--
-- Changes:
--   1. Make article_id nullable (was NOT NULL)
--   2. Add source_id UUID column (nullable, intentionally no FK — sources may be deleted)
--   3. Add CHECK constraint: at least one of article_id or source_id must be non-null
--   4. Add index on source_id for queue-processing lookups

-- Step 1: Drop the NOT NULL constraint on article_id
ALTER TABLE public.mutation_queue
    ALTER COLUMN article_id DROP NOT NULL;

-- Step 2: Add source_id column (no FK — sources can be hard-deleted)
ALTER TABLE public.mutation_queue
    ADD COLUMN IF NOT EXISTS source_id UUID DEFAULT NULL;

-- Step 3: Require at least one target to be non-null
DO $$ BEGIN
    ALTER TABLE public.mutation_queue
        ADD CONSTRAINT mutation_queue_target_check
        CHECK (article_id IS NOT NULL OR source_id IS NOT NULL);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Step 4: Index for source-pipeline lookups
CREATE INDEX IF NOT EXISTS idx_mutation_queue_source
    ON public.mutation_queue (source_id)
    WHERE source_id IS NOT NULL;
