import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding,Dropout,GRU
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import load_model

model_path = './luizgonzaga_gen_mjr.h5'
model = load_model('./luizgonzaga_gen_mjr.h5')

# Length of the vocabulary in chars
vocab = ['\n', ' ', '!', '"', "'", '(', ')', ',', '-', '.', '0', '2', '3', '5', '7', ':', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '}', 'À', 'É', 'Ê', 'Í', 'Ó', 'Ô', 'à', 'á', 'â', 'ã', 'ç', 'é', 'ê', 'í', 'ó', 'ô', 'õ', 'ú', 'ü', '’', '\ufeff']

def generate_text(model, start_seed,gen_size=100,temp=1.0):
    '''
    model: Trained Model to Generate Text
    start_seed: Intial Seed text in string form
    gen_size: Number of characters to generate

    Basic idea behind this function is to take in some seed text, format it so
    that it is in the correct shape for our network, then loop the sequence as
    we keep adding our own predicted characters. Similar to our work in the RNN
    time series problems.
    '''
    ind_to_char = np.array(vocab)
    char_to_ind = {char:ind for ind, char in enumerate(vocab)}

    # Number of characters to generate
    num_generate = gen_size

    # Vectorizing starting seed text
    input_eval = [char_to_ind[s] for s in start_seed]

    # Expand to match batch format shape
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty list to hold resulting generated text
    text_generated = []

    temperature = temp

    # Here batch size == 1
    model.reset_states()

    for i in range(num_generate):
        # Generate Predictions
        predictions = model(input_eval)

        # Remove the batch shape dimension
        predictions = tf.squeeze(predictions, 0)

        # Use a cateogircal disitribution to select the next character
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # Pass the predicted charracter for the next input
        input_eval = tf.expand_dims([predicted_id], 0)

        # Transform back to character letter
        text_generated.append(ind_to_char[predicted_id])
        print(text_generated)

    return (start_seed + ''.join(text_generated))
