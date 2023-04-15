# BabyAGI JS

BabyAGI JS is a JavaScript-based AI agent that creates, prioritizes, and executes tasks using the GPT-4 architecture. It integrates with OpenAI's language model to create a powerful AI that can handle a wide range of tasks.

## Features

- Task creation: Generates new tasks based on the current context and objectives.
- Task prioritization: Reorders tasks according to their importance and relevance to the main objective.
- Task execution: Performs tasks and returns results.

## How to use

1. Clone this repository.
2. Install the required dependencies using `npm install`.
3. Write your code in the `src` directory.
4. Run your program with `npm run start`.

## Main Files

### `src/index.ts`

This file initializes the BabyAGI agent with the required configurations, including the language model and objective. It imports the `BabyAGI` class from `babyagi.js` and creates a new instance to perform tasks based on the given objective.

### `src/babyagi.ts`

This file contains the core implementation of the BabyAGI agent. It defines three main classes, `TaskCreationChain`, `TaskPrioritizationChain`, and `ExecutionChain`, which are responsible for creating, prioritizing, and executing tasks, respectively.

The `BabyAGI` class combines these three classes and provides methods to add tasks, print tasks, and execute tasks. The `call` method is the main entry point to start the agent's task processing loop.

## Example

The following is an example of how to use the BabyAGI agent:

1. Set the objective in `src/index.ts`:

```javascript
const OBJECTIVE = 'Integrate stripe in typescript';
```

2. Run the program with `npm run start`.
3. The BabyAGI agent will create, prioritize, and execute tasks based on the given objective.

## Contributing

We welcome contributions to improve BabyAGI JS. Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
