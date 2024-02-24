import { chatModel as llm, createVectorStore } from "./utils";
import { ChatPromptTemplate } from "langchain/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { StringOutputParser } from "langchain/schema/output_parser";
import { createRetrievalChain } from "langchain/chains/retrieval";

// Document Chain
const outputParser = new StringOutputParser();
const prompt = ChatPromptTemplate.fromTemplate(`
Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}`);

const documentChain = await createStuffDocumentsChain({
  llm,
  prompt,
  outputParser,
});

// Retrieval Chain
const vectorStore = await createVectorStore(
  // "https://digitalidentity.co.jp/blog/seo/abbout-sge.html"
  // "https://js.langchain.com/docs/get_started/quickstart"
  "https://js.langchain.com/docs/integrations/document_loaders/web_loaders/serpapi"
);
const retriever = vectorStore.asRetriever();
const retrievalChain = await createRetrievalChain({
  combineDocsChain: documentChain,
  retriever,
});

const res = await retrievalChain.invoke({
  input: `Question: serp API Loader を使用して何をしていますか？
  Answer:`,
});
console.log(res);
