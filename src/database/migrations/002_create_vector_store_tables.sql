-- Migration: Create vector store tables for RAG system
-- Description: Creates tables for storing research documents and their vector embeddings
-- Requires: pgvector extension

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create research_documents table
CREATE TABLE IF NOT EXISTS research_documents (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    chunk_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create document_chunks table with vector embeddings
CREATE TABLE IF NOT EXISTS document_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES research_documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(1536), -- OpenAI text-embedding-3-small dimension
    chunk_index INTEGER NOT NULL,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_research_documents_user_id ON research_documents(user_id);
CREATE INDEX IF NOT EXISTS idx_research_documents_created_at ON research_documents(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_documents_metadata ON research_documents USING GIN(metadata);

CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_user_id ON document_chunks(user_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_document_chunks_metadata ON document_chunks USING GIN(metadata);

-- Create RLS policies for research_documents
ALTER TABLE research_documents ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only access their own documents
CREATE POLICY "Users can access own research documents" ON research_documents
    FOR ALL USING (auth.uid() = user_id);

-- Policy: Service role can access all documents
CREATE POLICY "Service role can access all research documents" ON research_documents
    FOR ALL USING (auth.role() = 'service_role');

-- Create RLS policies for document_chunks
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only access their own chunks
CREATE POLICY "Users can access own document chunks" ON document_chunks
    FOR ALL USING (auth.uid() = user_id);

-- Policy: Service role can access all chunks
CREATE POLICY "Service role can access all document chunks" ON document_chunks
    FOR ALL USING (auth.role() = 'service_role');

-- Create function for similarity search
CREATE OR REPLACE FUNCTION match_document_chunks(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.7,
    match_count int DEFAULT 5,
    user_id_filter uuid DEFAULT NULL
)
RETURNS TABLE (
    id text,
    document_id text,
    content text,
    metadata jsonb,
    chunk_index int,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        dc.id,
        dc.document_id,
        dc.content,
        dc.metadata,
        dc.chunk_index,
        1 - (dc.embedding <=> query_embedding) AS similarity
    FROM document_chunks dc
    WHERE 
        (user_id_filter IS NULL OR dc.user_id = user_id_filter)
        AND 1 - (dc.embedding <=> query_embedding) > match_threshold
    ORDER BY dc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for research_documents
CREATE TRIGGER update_research_documents_updated_at
    BEFORE UPDATE ON research_documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Grant necessary permissions
GRANT USAGE ON SCHEMA public TO authenticated, anon;
GRANT ALL ON research_documents TO authenticated, anon;
GRANT ALL ON document_chunks TO authenticated, anon;
GRANT EXECUTE ON FUNCTION match_document_chunks TO authenticated, anon;

-- Create indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_research_documents_type ON research_documents((metadata->>'type'));
CREATE INDEX IF NOT EXISTS idx_research_documents_topic ON research_documents((metadata->>'topic'));
CREATE INDEX IF NOT EXISTS idx_document_chunks_type ON document_chunks((metadata->>'type'));

-- Add comments for documentation
COMMENT ON TABLE research_documents IS 'Stores research documents and their metadata for RAG system';
COMMENT ON TABLE document_chunks IS 'Stores document chunks with vector embeddings for similarity search';
COMMENT ON FUNCTION match_document_chunks IS 'Performs similarity search on document chunks using cosine similarity';

-- Create view for document statistics
CREATE OR REPLACE VIEW document_stats AS
SELECT 
    rd.user_id,
    COUNT(rd.id) as total_documents,
    SUM(rd.chunk_count) as total_chunks,
    COUNT(CASE WHEN rd.metadata->>'type' = 'research_result' THEN 1 END) as research_documents,
    MAX(rd.created_at) as last_document_created
FROM research_documents rd
GROUP BY rd.user_id;

COMMENT ON VIEW document_stats IS 'Provides statistics about documents per user';

-- Grant access to the view
GRANT SELECT ON document_stats TO authenticated, anon;
