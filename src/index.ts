import * as dotenv from "dotenv";
import { BabyAGI } from "babyagi.js";
import { ChatOpenAI } from "langchain/chat_models";

dotenv.config();

const OBJECTIVE = 'Integrate stripe in typescript';

const llm = new ChatOpenAI({ temperature: 0 });

const babyAgi = BabyAGI.fromLLM(llm);

babyAgi.call({ objective: OBJECTIVE });
