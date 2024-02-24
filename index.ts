import mongoose from "mongoose";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { chatModel } from "./utils";
import { ChatPromptTemplate } from "langchain/prompts";

const Schema = mongoose.Schema;

const transcriptSchema = new Schema({
  title: {
    type: String,
    default: "",
  },
  subTitle: {
    type: String,
    default: "",
  },
  originalText: {
    type: [String],
    default: "",
  },
  translatedText: {
    type: [String],
    default: "",
  },
});

const Transcript = mongoose.model("Transcript", transcriptSchema);

mongoose
  // @ts-ignore
  .connect(Bun.env.MONGOURI)
  .then(() => console.log("MongoDB Connected"))
  .catch((err: unknown) => console.log(err));

const outputParser = new StringOutputParser();
const prompt = ChatPromptTemplate.fromMessages([
  ["system", "あなたはChatGPTのプロフェッショナルです。"],
  [
    "user",
    "以下の文章を翻訳してください。\n`link chain` という単語は LangChain と訳してください。\n{input}",
  ],
]);

const llmChain = prompt.pipe(chatModel).pipe(outputParser);

const transcripts = await Transcript.find({});

console.log(transcripts.length);

for (const ts of transcripts) {
  const id = ts._id;

  if (!ts.originalText[0]) continue;
  if (ts.subTitle) console.log(ts.subTitle);

  const res = await llmChain.invoke({
    input: ts.originalText[0],
  });
  console.log(res);
  await Transcript.findOneAndUpdate(
    { _id: id },
    { $set: { translatedText: [res] } }
  );

  // await Transcript();
}
