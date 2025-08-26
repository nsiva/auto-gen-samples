
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    embedding VECTOR(1536), -- Assuming OpenAI's text-embedding-ada-002 model which has 1536 dimensions
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Optional: Create an index for faster similarity search, especially for large datasets
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100); -- Adjust 'lists' based on your data size

--TODO later to enable admin permission
--CREATE POLICY "Allow admin to insert documents" ON documents
--FOR INSERT TO authenticated
--WITH CHECK (auth.role() = 'admin'); -- Assuming 'admin' is a role in your auth.users table or custom claims


CREATE POLICY "Allow authenticated users to view documents" ON documents
FOR SELECT TO authenticated
USING (true); -- Allows all authenticated users to read all documents


-- This example is for user-tied documents, not strictly requested but good for context
-- Make sure your documents table has a user_id UUID column referencing auth.users(id)
--CREATE POLICY "Users can view their own documents" ON documents
--FOR SELECT TO authenticated
--USING (auth.uid() = user_id);


CREATE EXTENSION IF NOT EXISTS vector;

CREATE OR REPLACE FUNCTION public.match_documents(query_embedding vector, match_threshold double precision DEFAULT 0.5, match_count integer DEFAULT 5, auth_uid uuid DEFAULT NULL::uuid)
 RETURNS TABLE(id uuid, content text, metadata jsonb, created_at timestamp with time zone, similarity double precision)
 LANGUAGE plpgsql
AS $function$
BEGIN
  RETURN QUERY
  SELECT
    d.id,
    d.content,
    d.metadata,
    d.created_at,
    (1 - (d.embedding <=> query_embedding)) AS similarity -- Cosine similarity between 0 and 1
  FROM
    documents d
  WHERE
    (d.embedding <=> query_embedding) < (1 - match_threshold)
    -- AND (auth_uid IS NULL OR d.user_id = auth_uid) -- Uncomment if documents are tied to users
  ORDER BY
    d.embedding <=> query_embedding
  LIMIT match_count;
END;
$function$
