import tensorflow as tf
from keras.models import load_model
from tensorflowjs.converters import save_keras_model


# Definir a função de perda personalizada
def sparse_cat_loss(y_true, y_pred):
    # Implemente sua função de perda aqui
    pass

# Registrar a função de perda personalizada
tf.keras.utils.get_custom_objects()['sparse_cat_loss'] = sparse_cat_loss

# Carregar o modelo salvo em formato H5
model = tf.keras.models.load_model('luizgonzaga_gen_mjr.h5')

# Converter e salvar o modelo em formato JSON compatível com o TensorFlow.js
save_keras_model(model, 'converted_model')