import { loadLayersModel, tensor2d } from "@tensorflow/tfjs";

import fetch from "node-fetch";

const setup = async () => {
    return {model: await loadModel(), metadata: await getMetaData()};
}

const getMessages = () => {
  const messages = [`die hard mario fan and i loved this game br br this game starts slightly boring but
  trust me it's worth it as soon as you start your
   hooked the levels are fun and exiting they will hook you OOV your mind turns to mush i'm not 
   kidding this game is also orchestrated and is beautifully done br br to keep this spoiler free i 
   have to keep my mouth shut about details but please try this
  game it'll be worth it br br story 9 9 action 10 1 it's that good OOV 10 attention OOV 10 average 10`,
 `the mother in this movie is reckless with her children to the point of neglect i wish i wasn't
  so angry about her and her actions because i would have otherwise enjoyed the flick what a number
   she was take my advise and fast forward through everything you see her do until the end also is anyone 
   else getting sick of watching movies that are filmed so dark anymore one can hardly 
   see what is being filmed as an audience we are impossibly involved with the 
   actions on the screen so then why the hell can't we have night vision`
]
 return messages;
}

const getMetaData = async () => {
  const metadata = await fetch("https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json")
  return metadata.json()
}

const padSequences = (sequences, metadata) => {
  return sequences.map(seq => {
    if (seq.length > metadata.max_len) {
      seq.splice(0, seq.length - metadata.max_len);
    }
    if (seq.length < metadata.max_len) {
      const pad = [];
      for (let i = 0; i < metadata.max_len - seq.length; ++i) {
        pad.push(0);
      }
      seq = pad.concat(seq);
    }
    return seq;
  });
}

const loadModel = async () => {
  const url = `https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json`;
  const model = await loadLayersModel(url);
  return model;
};

const predict = (text, model, metadata) => {
  const trimmed = text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' ');
  const sequence = trimmed.map(word => {
    const wordIndex = metadata.word_index[word];
    if (typeof wordIndex === 'undefined') {
      return 2; //oov_index
    }
    return wordIndex + metadata.index_from;
  });
  const paddedSequence = padSequences([sequence], metadata);
  const input = tensor2d(paddedSequence, [1, metadata.max_len]);

  const predictOut = model.predict(input);
  const score = predictOut.dataSync()[0];
  predictOut.dispose();
  return score;
}

const getSentiment = (score) => {
  if (score > 0.66) {
    return `Score of ${score} is Positive`;
  }
  else if (score > 0.4) {
    return `Score of ${score} is Neutral`;
  }
  else {
    return `Score of ${score} is Negative`;
  }
}

const run = async (text, modelData) => {
  let {model, metadata} = modelData;
  let sum = 0;
  text.forEach(function (prediction) {
    console.log(` ${prediction}`);
    let stPercentage = predict(prediction, model, metadata);
    let fPerc = parseFloat(stPercentage, 10);
    sum += fPerc;
    console.log('Sentiment for message');
    console.log(getSentiment(fPerc));
    console.log('============================');
  })
  console.log('\n');
  console.log('============================');

  console.log('Average sentiment');
  console.log(getSentiment(sum/text.length));
}

run(getMessages(), await setup());