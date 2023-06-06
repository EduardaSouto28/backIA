const tf = require('@tensorflow/tfjs-node');

const vocab = ['\n', ' ', '!', '"', "'", '(', ')', ',', '-', '.', '0', '2', '3', '5', '7', ':', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '}', 'À', 'É', 'Ê', 'Í', 'Ó', 'Ô', 'à', 'á', 'â', 'ã', 'ç', 'é', 'ê', 'í', 'ó', 'ô', 'õ', 'ú', 'ü', '’', '\ufeff']
const char_to_ind = {};
for (let i = 0; i < vocab.length; i++) {
  const char = vocab[i];
  char_to_ind[char] = i;
}

const ind_to_char = new Array(vocab.length);
for (let i = 0; i < vocab.length; i++) {
  ind_to_char[i] = vocab[i];
}

async function loadModel(modelPath) {
  return await tf.loadLayersModel(`file://${modelPath}`);
}

async function generateText(model, startSeed, genSize = 100, temp = 1.0) {
    // Número de caracteres a gerar
    const numGenerate = genSize;
  
    // Vetorizando o texto inicial de semente
    const inputEval = Array.from(startSeed, char => char_to_ind[char]);
  
    // Expande para corresponder ao formato do batch
    const inputEvalTensor = tf.tensor2d(inputEval, [1, inputEval.length]);
  
    // Lista vazia para armazenar o texto gerado
    const textGenerated = [];
  
    // Aqui, o tamanho do batch é 1
    model.resetStates();
  
    for (let i = 0; i < numGenerate; i++) {
      // Gerar previsões
      const predictions = model.predict(inputEvalTensor);
  
      // Remove a dimensão do formato do batch
      const predictionsSqueezed = predictions.squeeze();
  
      // Use uma distribuição categórica para selecionar o próximo caractere
      const predictionsNormalized = predictionsSqueezed.div(temp);
      const predictedIdTensor = tf.multinomial(predictionsNormalized, 1);
      const predictedId = predictedIdTensor.dataSync()[0];
  
      // Passa o caractere previsto para a próxima entrada
      inputEval.push(predictedId);
      inputEval.shift();
  
      // Transforma de volta para o caractere correspondente
      const predictedChar = ind_to_char[predictedId];
      textGenerated.push(predictedChar);
      console.log(textGenerated);
    }
  
    const generatedText = startSeed + textGenerated.join('');
    return generatedText;
}
  
module.exports = { loadModel, generateText, vocab, char_to_ind, ind_to_char };
