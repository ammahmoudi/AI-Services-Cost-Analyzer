-- Migration 011: Add purpose column to llm_configurations
-- This can be run directly on PostgreSQL

DO $$ 
BEGIN
    -- Check if column exists, if not add it
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name='llm_configurations' 
        AND column_name='purpose'
    ) THEN
        ALTER TABLE llm_configurations 
        ADD COLUMN purpose VARCHAR(50) DEFAULT 'extraction';
        
        RAISE NOTICE 'Added purpose column to llm_configurations';
    ELSE
        RAISE NOTICE 'Column purpose already exists in llm_configurations';
    END IF;
END $$;

-- Update existing records to have extraction purpose
UPDATE llm_configurations 
SET purpose = 'extraction' 
WHERE purpose IS NULL;
