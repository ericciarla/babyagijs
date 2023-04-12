import { OpenAI } from "langchain";
import { BabyAGI } from "babyagi.js";

const OBJECTIVE = 'Integrate stripe in typescript';

const llm = new OpenAI({ temperature: 0 });

const babyAgi = BabyAGI.fromLLM(llm, 'stripe');

babyAgi.call({ objective: OBJECTIVE });
