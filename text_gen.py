import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np 
import os
import time

"""
Text Generation with RNN
"""

PATH = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
EPOCHS = 10
BATCH_SIZE = 64
BUFFER_SIZE = 10000
EMBED_DIM = 256
RNN_UNITS = 512

def main():
    text = open(PATH, 'rb').read().decode(encoding = 'utf-8')
    print("opened")
    vocab, idx2char, char2idx, sequences = process_data(text)
    vocab_size = len(vocab)
    dataset = sequences.map(split_input_target) # map applies a function to all the input list
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder = True)

    model = build_model(vocab_size, BATCH_SIZE)

    # configure checkpoints
    checkpoint_dir, checkpoint_callback = configure(model, dataset, vocab_size)

    # train
    model.fit(dataset, epochs = EPOCHS, callbacks=[checkpoint_callback])

    # generate text
    # restore the latest checkpoint
    tf.train.latest_checkpoint(checkpoint_dir)
    gen_model = build_model(vocab_size, batch_size=1)
    gen_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    gen_model.build(tf.TensorShape([1, None]))
    gen_model.summary()

    generated_text = generate_text(gen_model, u"Hack: ", char2idx, idx2char)
    print(generated_text)

    return

def process_data(text):
    # get the unique characters
    vocab = sorted(set(text))

    # map characters to numbers and numbers to characters
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    # convert text to int
    text_as_int = np.array([char2idx[c] for c in text])

    # set each input seq length to be 100
    seq_length = 100
    examples_per_epoch = len(text)//(seq_length+1) # floor division

    # convert text vector into a stream of character indices
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    # the batch method converts individual characters to sequences of desired size
    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
    
    return (vocab, idx2char, char2idx, sequences)

# duplicate and shift it to form the input and target text by using the map methid
def split_input_target(chunk):
    input_text = chunk[:-1] # from the beginning to the last
    target_text = chunk[1:] # all characters except the last
    return input_text, target_text

def print_targets(dataset, idx2char):
    for input_example, target_example in dataset.take(2):
        print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
        print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))

    # each index is process at one time step
    for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
        print("Step {:4d}".format(i))
        print(" input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
        print(" expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))
    return

def build_model(vocab_size, batch_size):
    model = keras.Sequential()
    
    # the input layer
    model.add(layers.Embedding(vocab_size, EMBED_DIM, batch_input_shape = [batch_size, None]))

    # add three layers of LSTM with dropouts
    model.add(layers.LSTM(RNN_UNITS, return_sequences=True, stateful = True, input_shape = [batch_size, None]))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(RNN_UNITS, return_sequences=True, stateful = True, input_shape = [batch_size, None]))
    model.add(layers.Dropout(0.2))

    # the output layer
    model.add(layers.Dense(vocab_size))
    model.add(layers.Activation('softmax'))

    # add custom backward layer (change loss to cross-entropy gets some input dimension error)
    model.compile(loss=loss, optimizer='adam', metrics = ['accuracy'])

    return model

def try_model(dataset, idx2char, model):
    # check the shape of the output
    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "{batch_size, sequence_length, vocab_size}")
    
    model.summary()

    # to get actual predictions of the model, we sample from the output distribution
    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices, axis = -1).numpy()
    print(sampled_indices)

    print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
    print()
    print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))

    # check the example batch
    example_batch_loss = loss(target_example_batch, example_batch_predictions)
    print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    print("scalar_loss:      ", example_batch_loss.numpy().mean())

    return

# train the model
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def configure (model, dataset, vocab_size):
    # configure checkpoint
    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'

    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    
    return (checkpoint_dir, checkpoint_callback)

# set up the prediction loop
def generate_text(model, start_string, char2idx, idx2char):
    # number of characters to generate
    num_generate = 1000

    # converting start string to numbers
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # empty string to store results
    text_generated = []

    # set temp (high temp -> more random, low temp -> more predictable)
    temp = 0.5

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0) # remove all type of size 1

        # use categorical distribution to predict the character returned by the model
        predictions = predictions / temp
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # pass the predicted character as the next input to the model along with some previosu hidden state
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

if __name__ == "__main__":
    main()