import tensorflow as tf
from CNN_encoder import CNN_Encoder
from configs import argHandler
from tokenizer_wrapper import TokenizerWrapper
import time
from gpt2.gpt2_model import TFGPT2LMHeadModel
import numpy as np
from PIL import Image
import os

def preprocess_image(image_path, target_size=(224, 224)):
    """Loads and preprocesses a single image to be model-ready."""
    if not os.path.exists(image_path):
        print(f"!!! ERROR: Image not found at path: {image_path}!!!")
        return None
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize pixel values to 
    img_array = np.expand_dims(img_array, axis=0) # Add the batch dimension
    return tf.convert_to_tensor(img_array, dtype=tf.float32)

def generate_report(FLAGS, encoder, decoder, tokenizer_wrapper, image_tensor):
    """Generates a report for a single preprocessed image tensor."""
    visual_features, tags_embeddings = encoder(image_tensor, training=False)
    dec_input = tf.expand_dims(tokenizer_wrapper.GPT2_encode("startseq", pad=False), 0)
    
    num_beams = FLAGS.beam_width

    # The model expects the features to be tiled for each beam
    visual_features = tf.tile(visual_features, [num_beams, 1, 1])
    tags_embeddings = tf.tile(tags_embeddings, [num_beams, 1, 1])

    # Generate token sequence
    tokens = decoder.generate(dec_input, 
                              max_length=FLAGS.max_sequence_length, 
                              num_beams=num_beams, 
                              min_length=3,
                              eos_token_ids=tokenizer_wrapper.GPT2_eos_token_id(), 
                              no_repeat_ngram_size=0,
                              visual_features=visual_features,
                              tags_embedding=tags_embeddings, 
                              do_sample=False, 
                              early_stopping=True)

    # Decode the tokens into a sentence and clean it up
    tokens = tokens.numpy().tolist()
    sentence = tokenizer_wrapper.GPT2_decode(tokens[0])
    sentence = tokenizer_wrapper.filter_special_words(sentence)
    return sentence

if __name__ == "__main__":
    # --- 1. SET YOUR FILE PATHS HERE ---
    IMAGE_TO_TEST = "test_image3.png"
    CHECKPOINT_DIRECTORY = r"checkpoints\CDGPT2\best_ckpt"
    # ------------------------------------

    print("--- Initializing models and tokenizer ---")
    
    # Load default configurations from the project's config file
    FLAGS = argHandler()
    FLAGS.setDefaults()

    # Override the default checkpoint path with our specific one
    FLAGS.ckpt_path = CHECKPOINT_DIRECTORY

    # Initialize the tokenizer (this requires the original data CSV)
    try:
        tokenizer_wrapper = TokenizerWrapper(
            FLAGS.all_data_csv,
            FLAGS.csv_label_columns[0],  # ✅ get the first column name
            FLAGS.max_sequence_length,
            FLAGS.tokenizer_vocab_size
        )

    except FileNotFoundError:
        print(f"!!! ERROR: Could not find the data file '{FLAGS.all_data_csv}' needed to build the tokenizer.!!!")
        print("Please ensure you have run 'python get_iu_xray.py' first.")
        exit()


    # Initialize the models
    encoder = CNN_Encoder('pretrained_visual_model', FLAGS.visual_model_name, FLAGS.visual_model_pop_layers,
                          FLAGS.encoder_layers, FLAGS.tags_threshold, num_tags=len(FLAGS.tags))

    decoder = TFGPT2LMHeadModel.from_pretrained('distilgpt2', from_pt=True, resume_download=True)

    optimizer = tf.keras.optimizers.Adam()

    # Set up the checkpoint manager
    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.ckpt_path, max_to_keep=1)

    # Restore the model from the latest checkpoint in the directory
    if ckpt_manager.latest_checkpoint:
        print(f"--- Restoring model from: {ckpt_manager.latest_checkpoint} ---")
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print("--- Model restored successfully ---")
    else:
        print(f"!!! ERROR: No checkpoint found in '{CHECKPOINT_DIRECTORY}'. Please check the path.!!!")
        exit()

    # Load and process the image
    print(f"\n--- Loading and preprocessing image: {IMAGE_TO_TEST} ---")
    image_tensor = preprocess_image(IMAGE_TO_TEST)
    
    if image_tensor is not None:
        print("--- Image loaded. Starting report generation... ---")
        
        start_time = time.time()
        predicted_sentence = generate_report(FLAGS, encoder, decoder, tokenizer_wrapper, image_tensor)
        end_time = time.time()

        # Print the final result
        print("\n----------------- GENERATED REPORT -----------------")
        print(predicted_sentence)
        print("----------------------------------------------------")
        print(f"Time taken for generation: {end_time - start_time:.2f} seconds")