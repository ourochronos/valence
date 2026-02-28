-- Migration 007: Add pipeline_status column to sources
-- Issue #565: Unified source ingest pipeline

-- Add pipeline_status to sources table
ALTER TABLE public.sources
    ADD COLUMN IF NOT EXISTS pipeline_status TEXT DEFAULT 'pending' NOT NULL;

-- Add check constraint for valid statuses
DO $$ BEGIN
    ALTER TABLE public.sources ADD CONSTRAINT sources_pipeline_status_check
    CHECK (pipeline_status IN ('pending', 'indexed', 'complete', 'failed'));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Index for quick lookups of pending/failed sources
CREATE INDEX IF NOT EXISTS idx_sources_pipeline_status
    ON public.sources (pipeline_status);

-- Allow source_pipeline operation in mutation_queue
ALTER TABLE public.mutation_queue
    DROP CONSTRAINT IF EXISTS mutation_queue_operation_check;

ALTER TABLE public.mutation_queue
    ADD CONSTRAINT mutation_queue_operation_check
    CHECK (operation = ANY (ARRAY[
        'split'::text,
        'merge_candidate'::text,
        'recompile'::text,
        'decay_check'::text,
        'recompile_degraded'::text,
        'source_pipeline'::text
    ]));
