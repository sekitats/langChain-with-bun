import { ChatPromptTemplate } from "langchain/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { chatModel } from "./utils";

const outputParser = new StringOutputParser();
// 最もシンプルな例
// await chatModel.invoke('what is SGE?');

// テンプレートを使う

const prompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant."],
  ["user", "{input}"],
]);

const chain = prompt.pipe(chatModel).pipe(outputParser);
const res = await chain.invoke({ input: "SGEとは何ですか？" });
console.log(res);
