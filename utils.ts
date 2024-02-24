import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Document } from "@langchain/core/documents";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatPromptTemplate } from "langchain/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { LanguageModelLike } from "@langchain/core/language_models/base";
import { BasePromptTemplate } from "@langchain/core/prompts";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { BaseRetrieverInterface } from "langchain/schema/retriever";
import { RunnableInterface, RunnableSequence } from "langchain/schema/runnable";

// ChatModel
export const chatModel = new ChatOpenAI({
  openAIApiKey: Bun.env.OPENAI_API_KEY,
  temperature: 0,
});

// Document Loader
export const getDocs = async (
  url: string
): Promise<Document<Record<string, unknown>>[]> => {
  const loader = new CheerioWebBaseLoader(url);
  const docs = await loader.load();
  return docs;
};

export const getSplitDocs = async (
  docs: Document<Record<string, unknown>>[]
) => {
  const splitter = new RecursiveCharacterTextSplitter();
  const splitDocs = await splitter.splitDocuments(docs);
  return splitDocs;
};

export const createVectorStore = async (url: string) => {
  const docs = await getDocs(url);
  const splitDocs = await getSplitDocs(docs);
  const vectorStore = await getVectorStore(splitDocs);
  return vectorStore;
};

// Vector Store
export const getVectorStore = async (
  splitDocs: Document<Record<string, unknown>>[]
) => {
  const embeddings = new OpenAIEmbeddings();
  const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
  );
  return vectorStore;
};

// Prompt
// export const getPrompt = (context: string, input: string) => {
//   return ChatPromptTemplate.fromTemplate(`Answer the following question based only on the provided context:

//   <context>
//   {${context}}
//   </context>

//   Question: {${input}}`);
// };

// Document Chain
export const generateDocumentChain = async (
  llm: LanguageModelLike,
  prompt: BasePromptTemplate
) => {
  const documentChain = await createStuffDocumentsChain({
    llm,
    prompt,
  });
  return documentChain;
};

// Retrieval Chain
export const generateRetrievalChain = async (
  documentChain: any,
  retriever: any
) => {
  const retrievalChain = await createRetrievalChain({
    combineDocsChain: documentChain,
    retriever,
  });
  return retrievalChain;
};
