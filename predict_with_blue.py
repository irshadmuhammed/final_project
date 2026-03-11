import tensorflow as tf
from CNN_encoder import CNN_Encoder
from configs import argHandler
from tokenizer_wrapper import TokenizerWrapper
import time
from gpt2.gpt2_model import TFGPT2LMHeadModel
import numpy as np
from PIL import Image
import os

# --- Import evaluation libraries ---
from nltk.translate import bleu_score
import nltk.translate.gleu_score as gleu
from nlgeval import NLGEval


# --- Helper functions for prediction ---
def preprocess_image(image_path, target_size=(224, 224)):
    """Loads and preprocesses a single image to be model-ready."""
    if not os.path.exists(image_path):
        print(f"!!! ERROR: Image not found at path: {image_path} !!!")
        return None

    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return tf.convert_to_tensor(img_array, dtype=tf.float32)


def generate_report(FLAGS, encoder, decoder, tokenizer_wrapper, image_tensor):
    """Generates a report for a single preprocessed image tensor."""
    visual_features, tags_embeddings = encoder(image_tensor, training=False)
    dec_input = tf.expand_dims(tokenizer_wrapper.GPT2_encode("startseq", pad=False), 0)
    num_beams = FLAGS.beam_width

    visual_features = tf.tile(visual_features, [num_beams, 1, 1])
    tags_embeddings = tf.tile(tags_embeddings, [num_beams, 1, 1])

    tokens = decoder.generate(
        dec_input,
        max_length=FLAGS.max_sequence_length,
        num_beams=num_beams,
        min_length=3,
        eos_token_ids=tokenizer_wrapper.GPT2_eos_token_id(),
        no_repeat_ngram_size=0,
        visual_features=visual_features,
        tags_embedding=tags_embeddings,
        do_sample=False,
        early_stopping=True
    )

    # --- FIX: Convert TensorFlow tensor → list of integers ---
    tokens = tokens.numpy().tolist() if isinstance(tokens, tf.Tensor) else tokens
    if isinstance(tokens[0], list):   # In case of nested beams
        tokens = tokens[0]

    sentence = tokenizer_wrapper.GPT2_decode(tokens)
    sentence = tokenizer_wrapper.filter_special_words(sentence)
    return sentence


# --- Evaluation function ---
def get_evaluation_scores(hypothesis, references, testing_mode=False):
    """Calculates evaluation scores comparing hypothesis to references."""
    # GLEU scores (based on n-gram matches)
    gleu_scores = {
        "Gleu_1": gleu.corpus_gleu(references, hypothesis, min_len=1, max_len=1),
        "Gleu_2": gleu.corpus_gleu(references, hypothesis, min_len=1, max_len=2),
        "Gleu_3": gleu.corpus_gleu(references, hypothesis, min_len=1, max_len=3),
        "Gleu_4": gleu.corpus_gleu(references, hypothesis, min_len=1, max_len=4)
    }

    if testing_mode:
        # NLGEval expects plain strings
        hyp_str_list = [' '.join(h) for h in hypothesis]
        ref_str_list = [[' '.join(r)] for r in references]

        n = NLGEval()  # Lightweight init
        scores = n.compute_metrics(ref_list=ref_str_list, hyp_list=hyp_str_list)
    else:
        # BLEU scores (1–4 grams)
        scores = {
            "Bleu_1": bleu_score.corpus_bleu(references, hypothesis, weights=[1.0]),
            "Bleu_2": bleu_score.corpus_bleu(references, hypothesis, weights=[0.5, 0.5]),
            "Bleu_3": bleu_score.corpus_bleu(references, hypothesis, weights=[1/3, 1/3, 1/3]),
            "Bleu_4": bleu_score.corpus_bleu(references, hypothesis, weights=[0.25, 0.25, 0.25, 0.25])
        }

    # Combine GLEU + other scores
    for key, val in gleu_scores.items():
        scores[key] = val
    return scores


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # --- 1. CONFIGURE YOUR INPUTS HERE ---
    IMAGE_TO_TEST = "test_image.png"
    CHECKPOINT_DIRECTORY = r"checkpoints\CDGPT2\best_ckpt"

    # Ground truth report for evaluation
    GROUND_TRUTH_REPORT = (
        "the cardiac silhouette and mediastinum size are within normal limits. "
        "there is no pulmonary edema. there is no focal consolidation. "
        "there are no pleural effusions. there is no evidence of pneumothorax."
    )
    # -----------------------------------------

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
        print(f"!!! ERROR: Could not find '{FLAGS.all_data_csv}'. Run 'python get_iu_xray.py' first. !!!")
        exit()

    encoder = CNN_Encoder(
        'pretrained_visual_model',
        FLAGS.visual_model_name,
        FLAGS.visual_model_pop_layers,
        FLAGS.encoder_layers,
        FLAGS.tags_threshold,
        num_tags=len(FLAGS.tags)
    )

    decoder = TFGPT2LMHeadModel.from_pretrained('distilgpt2', from_pt=True, resume_download=True)

    optimizer = tf.keras.optimizers.Adam()
    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.ckpt_path, max_to_keep=1)

    if ckpt_manager.latest_checkpoint:
        print(f"--- Restoring model from: {ckpt_manager.latest_checkpoint} ---")
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print("--- Model restored successfully ---")
    else:
        print(f"!!! ERROR: No checkpoint found in '{CHECKPOINT_DIRECTORY}'. !!!")
        exit()

    print(f"\n--- Loading and preprocessing image: {IMAGE_TO_TEST} ---")
    image_tensor = preprocess_image(IMAGE_TO_TEST)
    if image_tensor is None:
        exit()

    print("--- Generating report from image... ---")
    predicted_sentence = generate_report(FLAGS, encoder, decoder, tokenizer_wrapper, image_tensor)

    print("\n----------------- PREDICTION & EVALUATION -----------------")
    print(f"Generated Report (Hypothesis):\n{predicted_sentence}")
    print(f"\nGround Truth (Reference):\n{GROUND_TRUTH_REPORT}")
    print("----------------------------------------------------------")

    # Prepare data for evaluation function
    hypothesis_tokens = [predicted_sentence.lower().split()]
    reference_tokens = [GROUND_TRUTH_REPORT.lower().split()]

    # Get the scores
    print("\n--- Calculating evaluation scores... ---")
    evaluation_results = get_evaluation_scores(hypothesis_tokens, reference_tokens, testing_mode=True)

    print("\n-------------------- FINAL SCORES --------------------")
    for metric, score in evaluation_results.items():
        print(f"{metric:20s}: {score:.4f}")
    print("----------------------------------------------------")
