// 

const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const { loadModel, generateText, vocab, char_to_ind, ind_to_char } = require('./teste');

const app = express();
const port = process.env.PORT || 3000;

// Carregar o modelo
const model = loadModel('./luizgonzaga_gen_mjr.h5');

app.get('/generate-text', (req, res) => {
  const startSeed = req.query.seed;
//   const genSize = Number(req.query.size);
//   const temp = Number(req.query.temp);

  // Gerar o texto usando a função generate_text
  const generatedText = generateText(model, startSeed, genSize, temp);

  res.json({ generatedText });
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
