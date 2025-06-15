import argparse
from typing import List

import nltk
import numpy as np
from Levenshtein import distance as levenshtein_distance
from medvqa.datasets.chest_imagenome import (
    CHEST_IMAGENOME_BBOX_NAMES,
    CHEST_IMAGENOME_LARGE_FAST_CACHE_DIR,
)
from medvqa.models.huggingface_utils import (
    CachedTextEmbeddingExtractor,
    SupportedHuggingfaceMedicalBERTModels,
)
from medvqa.utils.files_utils import (
    get_file_path_with_hashing_if_too_long,
    load_jsonl,
    save_pickle,
)
from medvqa.utils.logging_utils import print_blue, print_bold
from medvqa.utils.math_utils import rank_vectors_by_dot_product
from nltk.translate.bleu_score import sentence_bleu


def _levenshtein_similarity(sent_a: str, sent_b: str):
    """Computes Levenshtein similarity between two strings."""
    max_len = max(len(sent_a), len(sent_b))
    if max_len == 0:
        return 1.0
    return 1 - levenshtein_distance(sent_a, sent_b) / max_len


def _bleu1_similarity(tok_sent_a: List[str], tok_sent_b: List[str]) -> float:
    """Computes BLEU-1 score between two tokenized sentences."""
    return sentence_bleu([tok_sent_a], tok_sent_b, weights=(1.0,))


def _hybrid_score_similarity(
    sent_a: str,
    tok_sent_a: List[str],
    sent_embd_a: np.ndarray,
    sent_b: str,
    tok_sent_b: List[str],
    sent_embd_b: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    """
    Computes a hybrid similarity score combining semantic, BLEU-1, and
    Levenshtein similarities.
    """
    # Assumes sent_embd_a and sent_embd_b are normalized
    semantic_sim = np.dot(sent_embd_a, sent_embd_b)
    bleu1_sim_val = _bleu1_similarity(tok_sent_a, tok_sent_b)
    lev_sim_val = _levenshtein_similarity(sent_a, sent_b)
    return alpha * semantic_sim + beta * bleu1_sim_val + gamma * lev_sim_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=SupportedHuggingfaceMedicalBERTModels.get_all(),
    )
    parser.add_argument(
        "--device", type=str, default="GPU", choices=["CPU", "GPU"]
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_checkpoint_folder_path", type=str, required=True)
    parser.add_argument("--include_top_k_most_similar", action="store_true")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--integrated_fact_metadata_filepath", type=str, default=None
    )
    parser.add_argument(
        "--hybrid_alpha",
        type=float,
        default=0.5,
        help="Weight for the semantic similarity in the hybrid score.",
    )
    parser.add_argument(
        "--hybrid_beta",
        type=float,
        default=0.3,
        help="Weight for the BLEU-1 similarity in the hybrid score.",
    )
    parser.add_argument(
        "--hybrid_gamma",
        type=float,
        default=0.2,
        help="Weight for the Levenshtein similarity in the hybrid score.",
    )

    args = parser.parse_args()

    if args.include_top_k_most_similar:
        assert (
            args.top_k > 0
        ), "top_k must be greater than 0 if include_top_k_most_similar is True"
        assert (
            args.integrated_fact_metadata_filepath is not None
        ), "integrated_fact_metadata_filepath must be provided if include_top_k_most_similar is True"
        # Download punkt tokenizer data if not already present
        try:
            nltk.data.find("tokenizers/punkt")
        except nltk.downloader.DownloadError:
            print("Downloading NLTK's 'punkt' tokenizer...")
            nltk.download("punkt")

    # Define bbox phrases
    bbox_phrases = CHEST_IMAGENOME_BBOX_NAMES

    # Create embedding extractor
    embedding_extractor = CachedTextEmbeddingExtractor(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_checkpoint_folder_path=args.model_checkpoint_folder_path,
    )

    if args.include_top_k_most_similar:
        # Load integrated fact metadata
        print(
            "Loading integrated fact metadata from:",
            args.integrated_fact_metadata_filepath,
        )
        integrated_fact_metadata = load_jsonl(
            args.integrated_fact_metadata_filepath
        )

        # Extract anatomical locations
        anatomical_locations = set()
        for x in integrated_fact_metadata:
            al = x["metadata"]["anatomical location"]
            if al:
                anatomical_locations.add(al)
        anatomical_locations = list(anatomical_locations)
        print(f"Loaded {len(anatomical_locations)} anatomical locations.")

        # Extract embeddings
        print_bold("Extracting embeddings...")
        anatomical_location_embeddings = (
            embedding_extractor.compute_text_embeddings(anatomical_locations)
        )
        bbox_phrase_embeddings = embedding_extractor.compute_text_embeddings(
            bbox_phrases
        )
        most_similar_anatomical_locations = []
        most_similar_anatomical_location_embeddings = []
        most_similar_similarities = []

        print_bold("Ranking and re-ranking anatomical locations...")
        for i in range(len(bbox_phrase_embeddings)):
            # 1. Get top_k * 3 candidates based on embedding similarity
            idxs = rank_vectors_by_dot_product(
                anatomical_location_embeddings, bbox_phrase_embeddings[i]
            )
            prelim_top_k = args.top_k * 3
            prelim_idxs = idxs[:prelim_top_k]

            # 2. Re-rank these candidates using the hybrid score
            bbox_phrase = bbox_phrases[i]
            tok_bbox_phrase = nltk.word_tokenize(bbox_phrase)
            bbox_phrase_embd = bbox_phrase_embeddings[i]

            hybrid_scores = []
            for j in prelim_idxs:
                al_text = anatomical_locations[j]
                tok_al_text = nltk.word_tokenize(al_text)
                al_embd = anatomical_location_embeddings[j]
                score = _hybrid_score_similarity(
                    sent_a=bbox_phrase,
                    tok_sent_a=tok_bbox_phrase,
                    sent_embd_a=bbox_phrase_embd,
                    sent_b=al_text,
                    tok_sent_b=tok_al_text,
                    sent_embd_b=al_embd,
                    alpha=args.hybrid_alpha,
                    beta=args.hybrid_beta,
                    gamma=args.hybrid_gamma,
                )
                hybrid_scores.append((score, j))

            # Sort by hybrid score in descending order
            hybrid_scores.sort(key=lambda x: x[0], reverse=True)

            # 3. Select the final top_k from the re-ranked list
            final_top_k_idxs = [idx for score, idx in hybrid_scores[: args.top_k]]

            top_k_embeddings = anatomical_location_embeddings[final_top_k_idxs]
            # Use original dot product for the "similarities" value
            top_k_similarities = np.dot(top_k_embeddings, bbox_phrase_embd)

            most_similar_anatomical_locations.append(
                [anatomical_locations[idx] for idx in final_top_k_idxs]
            )
            most_similar_anatomical_location_embeddings.append(top_k_embeddings)
            most_similar_similarities.append(top_k_similarities)

        # Define output to save
        output = {
            "bbox_phrases": bbox_phrases,
            "bbox_phrase_embeddings": bbox_phrase_embeddings,
            "most_similar_anatomical_locations": most_similar_anatomical_locations,
            "most_similar_anatomical_location_embeddings": most_similar_anatomical_location_embeddings,
            "most_similar_similarities": most_similar_similarities,
        }
        save_path = get_file_path_with_hashing_if_too_long(
            folder_path=CHEST_IMAGENOME_LARGE_FAST_CACHE_DIR,
            prefix="bbox_phrase_embeddings_hybrid",
            strings=[
                args.model_name,
                args.model_checkpoint_folder_path,
                args.integrated_fact_metadata_filepath,
                f"top_k={args.top_k}",
                f"alpha={args.hybrid_alpha}",
                f"beta={args.hybrid_beta}",
                f"gamma={args.hybrid_gamma}",
            ],
            force_hashing=True,
        )
    else:
        # Extract embeddings
        bbox_phrase_embeddings = embedding_extractor.compute_text_embeddings(
            bbox_phrases
        )

        # Define output to save
        output = {
            "bbox_phrases": bbox_phrases,
            "bbox_phrase_embeddings": bbox_phrase_embeddings,
        }
        save_path = get_file_path_with_hashing_if_too_long(
            folder_path=CHEST_IMAGENOME_LARGE_FAST_CACHE_DIR,
            prefix="bbox_phrase_embeddings",
            strings=[
                args.model_name,
                args.model_checkpoint_folder_path,
            ],
            force_hashing=True,
        )

    # Save output
    print_blue("Saving output to:", save_path, bold=True)
    save_pickle(output, save_path)
    print("Done!")


if __name__ == "__main__":
    main()