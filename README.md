# BabyAGI JS (QA Branch)

# Features
- Supabase Integration (To get document context to help respond tasks in Execution chain)

# How to use
- Create a project and run this code in Supabase
```
-- Enable the pgvector extension to work with embedding vectors
create extension vector;

-- Create a table to store your documents
create table documents (
  id bigserial primary key,
  content text, -- corresponds to Document.pageContent
  metadata jsonb, -- corresponds to Document.metadata
  embedding vector(1536) -- 1536 works for OpenAI embeddings, change if needed
);

-- Create search function
CREATE OR REPLACE FUNCTION match_documents(
  table_name text,
  query_embedding double precision[],
  similarity_threshold double precision,
  match_count integer
)
RETURNS TABLE (
  content text,
  similarity double precision
)
LANGUAGE plpgsql
AS $$
DECLARE
  query_string text;
BEGIN
  query_string := 'SELECT content, 1 - (embedding <=> ' || quote_literal(query_embedding) || ') AS similarity
                   FROM ' || table_name || '
                   WHERE 1 - (embedding <=> ' || quote_literal(query_embedding) || ') > ' || quote_literal(similarity_threshold) || '
                   ORDER BY embedding <=> ' || quote_literal(query_embedding) || '
                   LIMIT ' || match_count || ';';
  RETURN QUERY EXECUTE query_string;
END;
$$;

-- Create an index to be used by the search function
create index on documents
  using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);
```

- Add and embed documents with context with your objective
- Clone this repository
- `npm install`
- Write your code in `src`
- `turbo run build lint check` to run build scripts quickly in parallel
- `npm start` to run your program