import * as dotenv from "dotenv";
import { OpenAI } from "langchain";
import { BabyAGI } from "babyagi.js";
import { OpenAIEmbeddings } from "langchain/embeddings";
import { SupabaseVectorStore } from "langchain/vectorstores";
import { SupabaseClient, createClient } from '@supabase/supabase-js';

dotenv.config();

const SUPABASE_URL = process.env.SUPABASE_URL || '';
const SUPABASE_API_KEY = process.env.SUPABASE_SERVICE_TOKEN || '';
const OBJECTIVE = 'Integrate stripe in typescript';


const llm = new OpenAI({ temperature: 0 });
const supabase: SupabaseClient = createClient(SUPABASE_URL, SUPABASE_API_KEY);

const supabase_vectorstore = new SupabaseVectorStore(new OpenAIEmbeddings(),{
  client: supabase,
  tableName: "documents",
  queryName: "match_documents",
});

const babyAgi = BabyAGI.fromLLM(llm, supabase_vectorstore);

babyAgi.call({ objective: OBJECTIVE });
