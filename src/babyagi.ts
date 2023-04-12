import * as dotenv from 'dotenv';
import { LLMChain, PromptTemplate } from "langchain";
import { BaseLLM } from "langchain/llms";
import { SupabaseClient, createClient } from '@supabase/supabase-js';
import { OpenAIEmbeddings } from 'langchain/embeddings';

dotenv.config();
const SUPABASE_URL = process.env.SUPABASE_URL || '';
const SUPABASE_API_KEY = process.env.SUPABASE_SERVICE_TOKEN || '';

class TaskCreationChain extends LLMChain {
    constructor(prompt: PromptTemplate, llm: BaseLLM) {
        super({prompt, llm});
    }

    static from_llm(llm: BaseLLM): LLMChain {
        const taskCreationTemplate: string =
            "You are a task creation AI that uses the result of an execution agent" +
            " to create new tasks with the following objective: {objective}," +
            " The last completed task has the result: {result}." +
            " This result was based on this task description: {task_description}." +
            " These are incomplete tasks: {incomplete_tasks}." +
            " Based on the result, create new tasks to be completed" +
            " by the AI system that do not overlap with incomplete tasks." +
            " Return the tasks as an array.";

        const prompt = new PromptTemplate({
          template: taskCreationTemplate,
          inputVariables: ["result", "task_description", "incomplete_tasks", "objective"],
        });

        return new TaskCreationChain(prompt, llm);
    }
}

class TaskPrioritizationChain extends LLMChain {
    constructor(prompt: PromptTemplate, llm: BaseLLM) {
        super({ prompt, llm});
    }

    static from_llm(llm: BaseLLM): TaskPrioritizationChain {
        const taskPrioritizationTemplate: string = (
            "You are a task prioritization AI tasked with cleaning the formatting of and reprioritizing" +
            " the following tasks: {task_names}." +
            " Consider the ultimate objective of your team: {objective}." +
            " Do not remove any tasks. Return the result as a numbered list, like:" +
            " #. First task" +
            " #. Second task" +
            " Start the task list with number {next_task_id}."
        );
        const prompt = new PromptTemplate({
            template: taskPrioritizationTemplate,
            inputVariables: ["task_names", "next_task_id", "objective"],
        });
        return new TaskPrioritizationChain(prompt, llm);
    }
}

class ExecutionChain extends LLMChain {
  constructor(prompt: PromptTemplate, llm: BaseLLM) {
      super({prompt, llm});
  }

  static from_llm(llm: BaseLLM): LLMChain {
      const executionTemplate: string =
          "You are an AI who performs one task based on the following objective: {objective}." +
          " Take into account these previously completed tasks: {context}." +
          " Take into account this context from documents that will help you respond to this task: {docContent}." +
          " Your task: {task}." +
          " Response:";

      const prompt = new PromptTemplate({
          template: executionTemplate,
          inputVariables: ["objective", "context", "docContent", "task"],
      });

      return new ExecutionChain(prompt, llm);
  }
}

async function getNextTask(
  taskCreationChain: LLMChain,
  result: string,
  taskDescription: string,
  taskList: string[],
  objective: string
): Promise<any[]> {
  const incompleteTasks: string = taskList.join(", ");
  const response: string = await taskCreationChain.predict({
      result,
      task_description: taskDescription,
      incomplete_tasks: incompleteTasks,
      objective,
  });

  const newTasks: string[] = response.split("\n");

  return newTasks
      .filter((taskName) => taskName.trim())
      .map((taskName) => ({ task_name: taskName }));
}

interface Task {
  task_id: number;
  task_name: string;
}

async function prioritizeTasks(
    taskPrioritizationChain: LLMChain,
    thisTaskId: number,
    taskList: Task[],
    objective: string
): Promise<Task[]> {
    
    const next_task_id = thisTaskId + 1;
    const task_names = taskList.map(t => t.task_name).join(', ');
    const response = await taskPrioritizationChain.predict({ task_names, next_task_id, objective });
    const newTasks = response.split('\n');
    const prioritizedTaskList: Task[] = [];

    for (const taskString of newTasks) {
        if (!taskString.trim()) {
            // eslint-disable-next-line no-continue
            continue;
        }
        const taskParts = taskString.trim().split('. ', 2);
        if (taskParts.length === 2) {
            const task_id = parseInt(taskParts[0].trim(), 10);
            const task_name = taskParts[1].trim();
            prioritizedTaskList.push({ task_id, task_name });
        }
    }

    return prioritizedTaskList;
}

export async function retrieve_embeddings( 
  table: string,
  query: string,
  sem_thres: number,
  match_count = 2,
): Promise<string> {
  const supabase_service: SupabaseClient = createClient(SUPABASE_URL, SUPABASE_API_KEY);
  const embeddings = new OpenAIEmbeddings();
  const embedding = await embeddings.embedQuery(query);

  const resp = supabase_service.rpc(
      "match_documents",
      {
          "table_name": table,
          "query_embedding": embedding,
          "similarity_threshold": sem_thres,
          "match_count": match_count,
      },
  );

  const docs = (await resp).data;
  const final_docs = docs.map((doc: {content: any;}) => 
     ({ content: doc.content }));
  const joinedContent = final_docs.map((doc: { content: any; }) => doc.content).join('');
  return joinedContent;
  
}

async function executeTask(
    executionChain: LLMChain,
    objective: string,
    task: string,
    doneTasks: string[],
    table: string,
): Promise<string> {
  const context = doneTasks.join(', ');
  const docContent = await retrieve_embeddings(table, task, 0.5);
  console.log(docContent);
  return executionChain.predict({objective,context,docContent,task});
}

export class BabyAGI {
  taskList: Array<Task> = [];

  taskCreationChain: TaskCreationChain;

  taskPrioritizationChain: TaskPrioritizationChain;

  executionChain: ExecutionChain;

  tableName: string;

  taskIdCounter = 1;

  maxIterations = 3;



  constructor(taskCreationChain: TaskCreationChain, taskPrioritizationChain: TaskPrioritizationChain, executionChain: ExecutionChain, tableName: string) {
      this.taskCreationChain = taskCreationChain;
      this.taskPrioritizationChain = taskPrioritizationChain;
      this.executionChain = executionChain;
      this.tableName = tableName; 
      
  }

  addTask(task: Task) {
      this.taskList.push(task);
  }

  printTaskList() {
      console.log('\x1b[95m\x1b[1m\n*****TASK LIST*****\n\x1b[0m\x1b[0m');
      this.taskList.forEach(t => console.log(`${t.task_id}: ${t.task_name}`));
  }

  printNextTask(task: Task) {
      console.log('\x1b[92m\x1b[1m\n*****NEXT TASK*****\n\x1b[0m\x1b[0m');
      console.log(`${task.task_id}: ${task.task_name}`);
  }

  printTaskResult(result: string) {
      console.log('\x1b[93m\x1b[1m\n*****TASK RESULT*****\n\x1b[0m\x1b[0m');
      console.log(result);
  }

  getInputKeys(): string[] {
      return ['objective'];
  }

  getOutputKeys(): string[] {
      return [];
  }

  async call(inputs: Record<string, any>): Promise<Record<string, any>> {
      const {objective} = inputs;
      const firstTask = inputs.first_task || 'Make a todo list';
      this.addTask({ task_id: 1, task_name: firstTask });
      let numIters = 0;
      let loop = true;
      const doneTasks = [];

      while (loop) {
          if (this.taskList.length) {
              this.printTaskList();
              const task = this.taskList.shift()!;
              this.printNextTask(task);
              doneTasks.push(task.task_name);
              const result = await executeTask(this.executionChain, objective, task.task_name, doneTasks, this.tableName);
              const thisTaskId = task.task_id;
              this.printTaskResult(result);
              
              const newTasks = await getNextTask(this.taskCreationChain, result, task.task_name, this.taskList.map(t => t.task_name), objective);
              

              (await newTasks).forEach(newTask => {
                this.taskIdCounter += 1;
                // eslint-disable-next-line no-param-reassign
                newTask.task_id = this.taskIdCounter;
                this.addTask(newTask);
              });
            
              this.taskList = await prioritizeTasks(this.taskPrioritizationChain, thisTaskId, this.taskList, objective);
            

              
          }
          numIters += 1;
          if (this.maxIterations !== null && numIters === this.maxIterations) {
            
              console.log('\x1b[91m\x1b[1m\n*****TASK ENDING*****\n\x1b[0m\x1b[0m');
              console.log(this.maxIterations);
              
              loop = false;
          }
      }

      return {};
  }

  static fromLLM(llm: BaseLLM, table_name:string): BabyAGI {
    const taskCreationChain = TaskCreationChain.from_llm(llm);
    const taskPrioritizationChain = TaskPrioritizationChain.from_llm(llm);
    const executionChain = ExecutionChain.from_llm(llm);
    return new BabyAGI(taskCreationChain, taskPrioritizationChain, executionChain, table_name);
    }
}



