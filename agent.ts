import { chatModel as llm, createVectorStore } from "./utils";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { ChatPromptTemplate } from "langchain/prompts";
import { pull } from "langchain/hub";
import { AgentExecutor, createOpenAIFunctionsAgent } from "langchain/agents";

const searchTool = new TavilySearchResults();
const agentPrompt = await pull<ChatPromptTemplate>(
  "hwchase17/openai-functions-agent"
);

const agent = await createOpenAIFunctionsAgent({
  llm,
  tools: [searchTool],
  prompt: agentPrompt,
});

const agentExecutor = new AgentExecutor({
  agent,
  tools: [searchTool],
  verbose: true,
});

const agentResult = await agentExecutor.invoke({
  input: `Question: 
  Answer:
  `,
});

console.log(agentResult.output);
