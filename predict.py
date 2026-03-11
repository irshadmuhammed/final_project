import tensorflow as tf
from CNN_encoder import CNN_Encoder
from configs import argHandler
from tokenizer_wrapper import TokenizerWrapper
import time
from gpt2.gpt2_model import TFGPT2LMHeadModel
import numpy as np
from PIL import Image
import os


# ---------------- IMAGE PREPROCESSING ----------------
def preprocess_image(image_path, target_size=(224, 224)):
    if not os.path.exists(image_path):
        print(f"!!! ERROR: Image not found at path: {image_path} !!!")
        return None

    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return tf.convert_to_tensor(img_array, dtype=tf.float32)


# ---------------- TEXT POST-PROCESSING (SAFETY NET) ----------------
def deduplicate_sentences(text):
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    unique_sentences = list(dict.fromkeys(sentences))
    return ". ".join(unique_sentences) + "."


# ---------------- REPORT GENERATION ----------------
def generate_report(FLAGS, encoder, decoder, tokenizer_wrapper, image_tensor):

    visual_features, tags_embeddings, _ = encoder(image_tensor, training=False)

    dec_input = tf.expand_dims(
        tokenizer_wrapper.GPT2_encode("startseq", pad=False), 0
    )

    num_beams = FLAGS.beam_width

    visual_features = tf.tile(visual_features, [num_beams, 1, 1])
    tags_embeddings = tf.tile(tags_embeddings, [num_beams, 1, 1])

    # 🔥 FIXED GENERATION CONFIG
    tokens = decoder.generate(
        dec_input,
        max_length=80,                          # ✅ limit length
        num_beams=num_beams,
        min_length=5,
        eos_token_ids=tokenizer_wrapper.GPT2_eos_token_id(),
        no_repeat_ngram_size=3,                 # ✅ prevent loops
        repetition_penalty=1.2,                 # ✅ discourage repeats
        visual_features=visual_features,
        tags_embedding=tags_embeddings,
        do_sample=False,
        early_stopping=True
    )

    tokens = tokens.numpy().tolist()
    sentence = tokenizer_wrapper.GPT2_decode(tokens[0])
    sentence = tokenizer_wrapper.filter_special_words(sentence)

    # ✅ final safety cleanup
    sentence = deduplicate_sentences(sentence)

    return sentence


# ---------------- MAIN ----------------
if __name__ == "__main__":

    IMAGE_TO_TEST = "test_image.png"
    CHECKPOINT_DIRECTORY = r"checkpoints\CDGPT2\best_ckpt"

    print("--- Initializing models and tokenizer ---")

    FLAGS = argHandler()
    FLAGS.setDefaults()
    FLAGS.ckpt_path = CHECKPOINT_DIRECTORY

    try:
        tokenizer_wrapper = TokenizerWrapper(
            FLAGS.all_data_csv,
            FLAGS.csv_label_columns[0],
            FLAGS.max_sequence_length,
            FLAGS.tokenizer_vocab_size
        )
    except FileNotFoundError:
        print("!!! ERROR: Tokenizer CSV not found. Run get_iu_xray.py first !!!")
        exit()

    encoder = CNN_Encoder(
        "pretrained_visual_model",
        FLAGS.visual_model_name,
        FLAGS.visual_model_pop_layers,
        FLAGS.encoder_layers,
        FLAGS.tags_threshold,
        num_tags=len(FLAGS.tags)
    )

    decoder = TFGPT2LMHeadModel.from_pretrained(
        "distilgpt2",
        from_pt=True,
        resume_download=True
    )

    optimizer = tf.keras.optimizers.Adam()

    ckpt = tf.train.Checkpoint(
        encoder=encoder,
        decoder=decoder,
        optimizer=optimizer
    )

    ckpt_manager = tf.train.CheckpointManager(
        ckpt, FLAGS.ckpt_path, max_to_keep=1
    )

    if ckpt_manager.latest_checkpoint:
        print(f"--- Restoring model from: {ckpt_manager.latest_checkpoint} ---")
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print("--- Model restored successfully ---")
    else:
        print("!!! ERROR: No checkpoint found !!!")
        exit()

    print(f"\n--- Loading image: {IMAGE_TO_TEST} ---")
    image_tensor = preprocess_image(IMAGE_TO_TEST)

    if image_tensor is not None:
        print("--- Generating report ---")
        start_time = time.time()

        predicted_sentence = generate_report(
            FLAGS, encoder, decoder, tokenizer_wrapper, image_tensor
        )

        end_time = time.time()

        print("\n----------------- GENERATED REPORT -----------------")
        print(predicted_sentence)
        print("----------------------------------------------------")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
