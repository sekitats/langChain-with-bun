import mongoose from "mongoose";
const Schema = mongoose.Schema;

export const transcriptSchema = new Schema({
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
