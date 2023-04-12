# BabyAGI JS (Main Branch)



# Overview
- BabyAGI but in JS (Python version: https://github.com/yoheinakajima/babyagi)
- The system uses OpenAI and Supabase APIs to create, prioritize, and execute tasks.
- Built with Langchain and Supabase

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

-- Create a function to search for documents
create function match_documents (
  query_embedding vector(1536),
  match_count int
) returns table (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    content,
    metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Create an index to be used by the search function
create index on documents
  using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);
```
- Clone this repository
- `npm install`
- Make this your .env
```
SUPABASE_URL=
SUPABASE_SERVICE_TOKEN=
OPENAI_API_KEY=
```
- `npm start` to run your program
