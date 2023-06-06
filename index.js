const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const app = express();
const port = 3000;

app.use(express.json());
app.use(cors());

function generateText(startSeed) {
    const pythonProcess = spawn('python', [
        './generate_text.py',
        startSeed,
    ]);

  return new Promise((resolve, reject) => {
    let generatedText = '';

    pythonProcess.stdout.on('data', (data) => {
      generatedText += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error(data.toString());
      reject(new Error('An error occurred while generating text'));
    });

    pythonProcess.on('close', () => {
      resolve(generatedText);
    });
  });
}

app.post('/', async (req, res) => {
  const startSeed = 'Maria';
  console.log(startSeed)

  try {
    const generatedText = await generateText(startSeed);
    res.json({ generatedText });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
