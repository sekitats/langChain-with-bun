# LangChain

## LLMs

OpenAI のCompletions API（text-davinci-003）を使うには OpenAIクラスを使う

```py
from langchain.llms import OpenAI

llm = OpenAI(model_name="text-davinci-003", temperature=0)

result = llm('自己紹介をしてください')
print(result)
```

```ts
import { ChatOpenAI } from "@langchain/openai";

const chatModel = new ChatOpenAI({
  openAIApiKey: Bun.env.OPENAI_API_KEY
});
const res = await chatModel.invoke("自己紹介をしてください");
```

## Chat Models

OpenAI の Chat Complete API (gpt-3.5-turbo, gpt-4)は単に一つのテキストを入力するのではなく、チャット形式のやりとりを入力して応答を得る

```py
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

chat = ChatOpenAI(
  model_name="gpt-3.5-turbo",
  temperature=0,
  streaming=True # ストリーミングを有効にする
  callbacks=[StreamingStdOutCallbackHandler()] # コールバックを有効にする
)

messages = [
  SystemMessage(content="You are a helpful assistant."),
  HumanMessage(content="こんにちは！私はジョンと言います"),
  AIMessage(content="こんにちは、ジョンさん！どのようにお手伝いできますか？"),
  HumanMessage(content="私の名前がわかりますか？")
]

result = chant(messages)
print(result.content)
```

これをTSで書くと

```ts
import { ChatPromptTemplate } from "@langchain/core/prompts";

const chatModel = new ChatOpenAI({
  openAIApiKey: Bun.env.OPENAI_API_KEY,
    temperature: 0,
});
const res = await chatModel.invoke("自己紹介をしてください");
```

## Prompts

### PromptTemplate

```py
from langchain.prompts import PromptTemplate

template = """
以下の料理のレシピを考えてください

料理名: {dish}
"""

prompt = PromptTemplate(
  input_variables=["dish"],
  template=template
)

result = prompt.format(dish="カレー")
print(result)
```

### ChatPromptTemplate

```py
from langchain.prompts import (
  ChatPromptTemplate,
  PromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage

chat_prompt = ChatPromptTemplate.from_message([
  SystemMessagePromptTemplate.from_template("あなたは{country}料理のプロフェッショナルです。"),
  HumanMessagePromptTemplate.from_template("以下の料理のレシピを考えてください\n\n料理名: {dish}")
])

messages = chat_prompt.format_prompt(country="イギリス", dish="肉じゃが").to_messages()
```

TSで書く

chatModel と prompt をchainで繋ぐ

```ts
import { ChatPromptTemplate } from "@langchain/core/prompts";

const chatModel = new ChatOpenAI({ openAIApiKey: Bun.env.OPENAI_API_KEY });
const prompt = ChatPromptTemplate.fromMessages([
  ["system", "あなたは{country}料理のプロフェッショナルです。"],
  ["user", "以下の料理のレシピを考えてください\n\n料理名: {dish}"],
]);
const chain = prompt.pipe(chatModel);
const res = await chain.invoke({
  country: "イギリス",
  dish: "肉じゃが",
});
console.log(res);
```

### Output Parses 

```py
from pydantic import BaseModel, Field

class Recipe(BaseModel):
  ingredients: list[str] = Field(description="ingredients of the dish")
  steps: list[str] = Field(description="steps to make the dish")

from langchain.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=Recipe)
```

そのクラスの定義で出力してほしい時に使う

TSではシンプルに文字列で返してもらおう

```ts
import { StringOutputParser } from "@langchain/core/output_parsers";

const chatModel = new ChatOpenAI({ openAIApiKey: Bun.env.OPENAI_API_KEY });
const outputParser = new StringOutputParser();
const prompt = ChatPromptTemplate.fromMessages([
  ["system", "あなたは{country}料理のプロフェッショナルです。"],
  ["user", "以下の料理のレシピを考えてください\n\n料理名: {dish}"],
]);
const llmChain = prompt.pipe(chatModel).pipe(outputParser);
const res = await llmChain.invoke({
  country: "イギリス",
  dish: "肉じゃが",
});
```

## Chains

LLMChain: PromptTemplate, Language Model, OutputParser を繋ぐ

SimpleSequentialChain: Chain と Chain を繋ぐ

- OpenAIModerationChain
- LLMRequestsChain
- OpenAPIEndpointChain
- PALChain
- SQLDatabaseChain

## Memory

保留

## Data Connection

Data Connection の背景となる、RAG (Retrieval Augmented Generation)

READMEの内容を文脈に含めて質問する

```py
文脈を踏まえて質問に1文で回答してください。

文脈: """
<LangChainのREADMEの内容>
"""

質問: LangChainとは？
```

ただし、LLMにはトークン数の最大値の制限があるため、あらゆるデータを context に入れることはできません。

そこで、文書を OpenAI の Embeddings API などでベクトル化しておいて、入力にベクトルが近い文章を検索して context に含める手法があり RAG と呼ばれる。

- あらかじめ用意したデータベースから検索する
- Googleなどの検索エンジンかでWeb上から検索することも

Data Connection では、とくに Vector Store を使い、文章をベクトル化して保存しておいて、入力のテキストとベクトルの近い文章を検索して context に含めて使う方法が提供されている

### Document Loaders

`!pip install GitPython==3.1.36`

```py
form langchain.document_loaders import GitLoader

def file_filter(file_path):
  clone_url="https://github.com/langchain-ai/langchain",
  repo_path="./langchain",
  branch="master",
  file_filter=file_filter,


raw_docs = loader.load()
print(raw_docs)
```

**Document Loader** の種類はたくさんある！

### Document Transformers

Document Loader で読み込んだデータは `ドキュメント` と呼ぶ。ドキュメントに何らかの変換をかけるのが `Document Transformers`

例えば、ある程度の長さのチャンクに分割する
→ `CharacterTextTextSplitter`を使う

### Text Embedding models

```py
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
```

テキストをベクトル化してみる

```py
query = "AWSのS3からデータを読み込むためのDocumentLoaderはありますか"

vector= embeddings.embed_query(query)
print(vector)
```

### Vector Stores

保存先の Vector Store を準備して、ドキュメントをベクトル化して保存します。本書では Chroma というVector Storesを使う

```py
from langchain.vectorstores import Chroma

db = Chroma.from_documents(docs, embeddings)
```

### Retrievers

```py
retriever = db.as_retriever()
```

---

# TSで書く

`npm install cheerio` する

```ts
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";

const loader = new CheerioWebBaseLoader(
  "https://docs.smith.langchain.com/overview"
);

const docs = await loader.load();

console.log(docs.length); // 
console.log(docs[0].pageContent.length); // 45772 
```

文字数が多いので text splitter でチャンクに分けている

```ts
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

const splitter = new RecursiveCharacterTextSplitter();
const splitDocs = await splitter.splitDocuments(docs);
console.log(splitDocs.length);
console.log(splitDocs[0].pageContent.length);
```

そしたら、ベクトル化して Vector Store に保存する

```ts
import { OpenAIEmbeddings } from "@langchain/openai";

const embeddings = new OpenAIEmbeddings();
```

`simple in-memory demo vectorstore` を使う

```ts
import { MemoryVectorStore } from "langchain/vectorstores/memory";

const vectorstore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  embeddings
);
```

> このデータをベクターストアにインデックス化したので、検索チェーンを作成する。このチェーンは、入力された質問を受け取り、関連するドキュメントを検索し、それらのドキュメントを元の質問と一緒にLLMに渡し、元の質問に答えるように依頼します。

```ts
// ChatModel
const chatModel = new ChatOpenAI({ openAIApiKey: Bun.env.OPENAI_API_KEY });

// Document Loader
const loader = new CheerioWebBaseLoader(
  "https://docs.smith.langchain.com/overview"
);
const docs = await loader.load();

// Document Transformer
const splitter = new RecursiveCharacterTextSplitter();
const splitDocs = await splitter.splitDocuments(docs);

// Embeddings
const embeddings = new OpenAIEmbeddings();

// Vector Store
const vectorstore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  embeddings
);

const prompt =
  ChatPromptTemplate.fromTemplate(`Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}`);

const documentChain = await createStuffDocumentsChain({
  llm: chatModel,
  prompt,
});

documentChain.invoke({
  input: "what is LangSmith?",
  // やろうと思えば、ドキュメントを直接渡して自分たちで実行することもできる：
  // context: [
  //   new Document({
  //     pageContent:
  //       "LangSmith is a platform for building production-grade LLM applications.",
  //   }),
  // ],
});

const retriever = vectorstore.asRetriever();

const retrievalChain = await createRetrievalChain({
  combineDocsChain: documentChain,
  retriever,
});

const result = await retrievalChain.invoke({
  input: "what is LangSmith?",
});

console.log(result.answer);
```


## Conversational Retrieval Chain

- 検索方法は、最新の入力に対してだけ働くのではなく、むしろ全履歴を考慮に入れるべきである。
- 最終的なLLMチェーンも同様に、歴史全体を考慮に入れるべきである。

保留

## Agents

どんな処理を行うべきか、LLMに選択して動いてほしい場合がある

- APIを叩いて欲しかったり
- SQLを実行して欲しかったり

ツールとしては、Vector Store を使って特定分野のデータを検索して使わせることもできれば、Google などの検索エンジンのAPIを使わせたりすることもできる


### 例 zero-shot-react-description

```py
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chat_model import ChatOpenAI

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
tools = load_tools(["terminal"])
agent_chain = initialize_agent(
  tools, chat, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

result = agent_chain.run("sample_dataディレクトリにあるファイルの一覧を教えて")
print(result)
```

これは、ReAct というAgent
その他にも、Plan-and-Solve, Function Calling がある

#### Tools

Agent のツール例

- terminal
- Python_REPL
- google_search
- Wikipedia
- human

例えば次のように自作できる

```py
from langchain.tools import Tool

def my_super_func(param):
  return "42"

tools = [
    Tool.from_function(
        func=my_super_func,
        name="The_Answer",
        description="生命、宇宙、そして万物についての究極の疑問の答え"
    )
]
```


独自の Retriever Toolを作る

```ts
import { createRetrieverTool } from "langchain/tools/retriever";

const retrieverTool = await createRetrieverTool(retriever, {
  name: "langsmith_search",
  description:
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
});
```

`Tavily`: は LLMの検索に特化した検索APIで、優れた検索体験を提供する

```
export TAVILY_API_KEY=...
```


```ts
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";

const searchTool = new TavilySearchResults();
const tools = [retrieverTool, searchTool];
```

### Toolkits

### Function calling を使う OpenAI Functions Agent

Function Calling を使ったAgentsをしようすると動作が安定しやすい

```py
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = load_tools(["terminal"])
agent_chain = initialize_agent(tools, chat, agent=AgentType.OPENAI_FUNCTIONS)
result = agent_chain.run("sample_dataディレクトリにあるファイルの一覧を教えて")
print(result)
```

### 一度に複数ツールを使う OpenAI Multi Function Agent

DuckDuckGo をツールとして OpenAI Multi Function Agent を初期化する

```py
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = load_tools(['ddg-search'])
agent_chain = initialize_agent(tools, chat, agent=AgentType.OPENAI_MULTI_FUNCTIONS)

result = agent_chain.run("東京と大阪の天気を教えて")
print(result)
```

#### OPENAI_MULTI_FUNCTIONSは何が違うの？

この質問に対しては、あきらきにツールを2回実行する必要がある

ツールを直接 function Calling の関数とするのではなく、ツールをまとめた `tool_selection` という関数が用意される。

LLM は tool_selection という関数の引数として、使いたいツールの名前と引数をリストで同時に複数返す。

---

TSで書く

TAVILY_API_KEY を忘れずに

```ts
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { ChatOpenAI } from "@langchain/openai";
import type { ChatPromptTemplate } from "@langchain/core/prompts";
import { pull } from "langchain/hub";
import { createOpenAIFunctionsAgent, AgentExecutor } from "langchain/agents";

const tools = [new TavilySearchResults()];
const prompt = await pull<ChatPromptTemplate>(
  "hwchase17/openai-functions-agent"
);
const llm = new ChatOpenAI({ modelName: "gpt-3.5-turbo-1106", temperature: 0 });
const agent = await createOpenAIFunctionsAgent({ llm, tools, prompt });
const agentExecutor = new AgentExecutor({ agent, tools });
const res = await agentExecutor.invoke({ input: "東京の天気は？" });
console.log(res);
```


