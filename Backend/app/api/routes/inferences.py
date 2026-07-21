from fastapi import APIRouter, HTTPException, Body, Request
import inspect
import asyncio
import logging
import hashlib
import difflib
import re
import string
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from app.services.model_loader_service import (
    transcribe_whisper_base,
    transcribe_whisper_large,
    wave2vec,
    extract_whisper_embeddings,
    extract_wav2vec2_embeddings,
    reduce_dimensions,
    predict_emotion_wave2vec,
    extract_audio_frequency_features,
    transcribe_whisper_with_attention,
    predict_emotion_wave2vec_with_attention,
    transcribe_whisper_with_timestamps,
    extract_whisper_attention_pairs,
)
from app.services.custom_model_service import (
    ModelValidationError,
    is_custom_model,
    resolve_model,
    run_custom_inference,
)
from app.services.dataset_service import resolve_file
from app.core.redis import get_result, cache_result

router = APIRouter()

# Define paths
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
UPLOAD_DIR = Path("uploads")

# Dataset directories
DATASET_DIRS = {
    "common-voice": DATA_DIR / "common_voice_valid_dev",
    "ravdess": DATA_DIR / "ravdess_subset",
}
logger = logging.getLogger(__name__)


def get_session_id(request: Request) -> Optional[str]:
    """Extract session ID from request (optional for backwards compatibility)"""
    return getattr(request.state, 'sid', None)


MODEL_FUNCTIONS = {
    "whisper-base": transcribe_whisper_base,
    "whisper-large": transcribe_whisper_large,
    "wav2vec2": wave2vec,
}


@router.post("/inferences/run")
async def run_inference_endpoint(
    http_request: Request,
    request: dict = Body(..., example={
        "model": "whisper-base",
        "file_path": "/path/to/audio.wav",
        "dataset": "common-voice", 
        "dataset_file": "sample-001.mp3"
    })
):
    # Extract parameters from request body
    model = request.get("model")
    file_path = request.get("file_path")
    dataset = request.get("dataset")
    dataset_file = request.get("dataset_file")
    
    if not model:
        raise HTTPException(status_code=400, detail="Model is required")
    
    session_id = get_session_id(http_request)
    return await run_inference(model, file_path, dataset, dataset_file, session_id)


@router.post("/inferences/batch-check")
async def check_batch_cache(
    http_request: Request,
    request: dict = Body(..., example={
        "model": "whisper-base",
        "dataset": "common-voice",
        "files": ["sample-001.mp3", "sample-002.mp3"]
    })
):
    """Check which files in a batch already have cached predictions"""
    model = request.get("model")
    dataset = request.get("dataset") 
    files = request.get("files", [])
    
    if not model or not dataset:
        raise HTTPException(status_code=400, detail="Model and dataset are required")
    
    session_id = get_session_id(http_request)
    
    cached_results = {}
    missing_files = []
    
    for filename in files:
        try:
            # Resolve the file path
            resolved_path = resolve_file(dataset, filename, session_id)
            
            # Create cache key
            file_content_hash = hashlib.md5(str(resolved_path).encode()).hexdigest()
            cache_key = f"{model}_{file_content_hash}"
            
            # Check cache
            cached_result = await get_result(model, cache_key)
            if cached_result is not None:
                cached_results[filename] = cached_result.get("prediction", cached_result)
            else:
                missing_files.append(filename)
                
        except (FileNotFoundError, ValueError):
            # File doesn't exist or invalid dataset
            missing_files.append(filename)
    
    return {
        "cached_results": cached_results,
        "missing_files": missing_files,
        "cache_hit_rate": len(cached_results) / len(files) if files else 0
    }


async def run_inference(
    model: str,
    file_path: Optional[str] = None,
    dataset: Optional[str] = None,
    dataset_file: Optional[str] = None,
    session_id: Optional[str] = None,
):
    """Internal function for running inference - can be called directly or via HTTP endpoint"""
    logger.info(
        "inferences.run model=%s file_path=%s dataset=%s dataset_file=%s session_id=%s",
        model,
        file_path,
        dataset,
        dataset_file,
        session_id,
    )

    # Built-in models dispatch through MODEL_FUNCTIONS; user-registered ones
    # are referenced as `custom:<session_id>:<name>` and run through the
    # generic, config-driven path in custom_model_service.
    custom_spec = None
    func = MODEL_FUNCTIONS.get(model)
    if not func:
        if not is_custom_model(model):
            raise HTTPException(status_code=400, detail=f"Invalid model: {model}")
        try:
            custom_spec = await resolve_model(model)
        except (ModelValidationError, ValueError) as e:
            raise HTTPException(status_code=404, detail=str(e))

    resolved_path: Optional[Path] = None

    if file_path:
        resolved_path = Path(file_path)
    elif dataset and dataset_file:
        try:
            # Resolve using service (enforces allowed datasets and basename-only)
            resolved_path = resolve_file(dataset, dataset_file, session_id)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            # Unknown dataset or other
            raise HTTPException(status_code=404, detail=str(e))
    else:
        raise HTTPException(
            status_code=400,
            detail="Missing audio reference. Provide either 'file_path' or 'dataset' + 'dataset_file'.",
        )

    if not resolved_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {resolved_path}")

    # Create cache key based on model and file path
    file_content_hash = hashlib.md5(str(resolved_path).encode()).hexdigest()
    cache_key = f"{model}_{file_content_hash}"
    
    # Check if result is cached
    cached_result = await get_result(model, cache_key)
    if cached_result is not None:
        logger.info(f"Returning cached result for {resolved_path}")
        return cached_result.get("prediction", cached_result)

    # Detect if function is async or sync and call appropriately
    if custom_spec is not None:
        try:
            prediction = await asyncio.to_thread(run_custom_inference, custom_spec, str(resolved_path))
        except Exception as e:
            logger.exception("Custom model %s failed on %s", model, resolved_path)
            raise HTTPException(status_code=500, detail=f"Custom model inference failed: {e}")
    elif inspect.iscoroutinefunction(func):
        prediction = await func(str(resolved_path))
    else:
        prediction = await asyncio.to_thread(func, str(resolved_path))

    # Cache the result for future use (6 hours TTL)
    await cache_result(model, cache_key, {"prediction": prediction}, ttl=6*60*60)
    logger.info(f"Cached prediction for {resolved_path}")

    return prediction


@router.post("/inferences/whisper-batch")
async def batch_whisper_analysis(request: Request):
    """
    Get batch whisper transcripts from cache and analyze common terms
    """
    try:
        body = await request.json()
        filenames = body.get("filenames", [])
        model = body.get("model", "whisper-base")
        dataset = body.get("dataset")
        
        logger.info(f"batch_whisper_analysis called with: filenames={len(filenames)} files, dataset={dataset}, model={model}")
        
        if not filenames:
            raise HTTPException(status_code=400, detail="No filenames provided")
        
        if len(filenames) > 50:  # Limit batch size
            raise HTTPException(status_code=400, detail="Too many files. Maximum 50 files per batch.")
        
        # Process each file - try to get from cache first
        individual_transcripts = []
        all_words = []
        cached_count = 0
        missing_count = 0
        
        session_id = get_session_id(request)
        
        for filename in filenames:
            try:
                # Get file path and create cache key
                if dataset:
                    file_path = resolve_file(dataset, filename, session_id)
                else:
                    file_path = UPLOAD_DIR / filename
                    if not file_path.exists():
                        print(f"Warning: File not found: {file_path}")
                        missing_count += 1
                        continue
                
                # Create cache key (same as used in regular inference)
                file_content_hash = hashlib.md5(str(file_path).encode()).hexdigest()
                cache_key = f"{model}_{file_content_hash}"
                
                # Try to get from cache first
                cached_result = await get_result(model, cache_key)
                
                # If not found and this is a custom dataset with session mismatch, try alternative cache keys
                if cached_result is None and dataset and dataset.startswith('custom:'):
                    from app.services.custom_dataset_service import parse_custom_dataset_name
                    try:
                        session_id_from_name, dataset_name = parse_custom_dataset_name(dataset)
                        if session_id_from_name != session_id:
                            # Try cache key with the original session ID path
                            from app.services.custom_dataset_service import get_custom_dataset_manager
                            original_manager = get_custom_dataset_manager(session_id_from_name)
                            original_file_path = original_manager.resolve_file_path(dataset_name, filename)
                            original_hash = hashlib.md5(str(original_file_path).encode()).hexdigest()
                            original_cache_key = f"{model}_{original_hash}"
                            cached_result = await get_result(model, original_cache_key)
                            if cached_result is not None:
                                logger.info(f"Found cached result using original session path for {filename}")
                    except Exception as e:
                        logger.warning(f"Could not try alternative cache key for {filename}: {e}")
                
                transcript = None
                if cached_result is not None:
                    # Extract transcript from cached result
                    if isinstance(cached_result, dict):
                        transcript = cached_result.get("prediction", cached_result.get("transcript"))
                    else:
                        transcript = cached_result
                    cached_count += 1
                    logger.info(f"Using cached transcript for {filename}")
                else:
                    # Not in cache - run inference to generate transcript
                    logger.info(f"No cached transcript found for {filename}, running inference...")
                    try:
                        # Run inference to generate transcript
                        inference_result = await run_inference(model, None, dataset, filename, session_id)
                        
                        if inference_result and isinstance(inference_result, dict):
                            transcript = inference_result.get("prediction", inference_result.get("transcript"))
                            if transcript:
                                logger.info(f"Generated and cached transcript for {filename}")
                                # The transcript is automatically cached by run_inference
                            else:
                                logger.warning(f"No transcript in inference result for {filename}")
                                missing_count += 1
                                continue
                        elif isinstance(inference_result, str):
                            transcript = inference_result
                            logger.info(f"Generated transcript for {filename}")
                        else:
                            logger.warning(f"Invalid inference result for {filename}: {type(inference_result)}")
                            missing_count += 1
                            continue
                            
                    except Exception as inference_error:
                        logger.error(f"Failed to run inference for {filename}: {inference_error}")
                        missing_count += 1
                        continue
                
                if transcript:
                    # Clean and tokenize transcript
                    words = transcript.lower().split()
                    # Remove common stop words and punctuation
                    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
                    filtered_words = [word.strip('.,!?";:()[]{}').lower() for word in words if word.strip('.,!?";:()[]{}').lower() not in stop_words and len(word.strip('.,!?";:()[]{}')) > 2]
                    
                    individual_transcripts.append({
                        "filename": filename,
                        "transcript": transcript,
                        "word_count": len(words)
                    })
                    
                    all_words.extend(filtered_words)
                    
            except Exception as file_error:
                print(f"Error processing {filename}: {file_error}")
                missing_count += 1
                continue
        
        logger.info(f"Successfully processed {len(individual_transcripts)} files out of {len(filenames)} (cached: {cached_count}, generated: {len(individual_transcripts) - cached_count}, failed: {missing_count})")
        if not individual_transcripts:
            raise HTTPException(status_code=404, detail=f"Could not process any of the selected files. Failed to generate transcripts for {missing_count} files.")
        
        # Calculate word frequency
        from collections import Counter
        word_counts = Counter(all_words)
        total_words = len(all_words)
        
        # Get top terms with percentages
        common_terms = []
        for word, count in word_counts.most_common(10):  # Get top 10, frontend will show top 5
            percentage = (count / total_words) * 100
            common_terms.append({
                "term": word,
                "count": count,
                "percentage": percentage
            })
        
        return {
            "common_terms": common_terms,
            "individual_transcripts": individual_transcripts,
            "summary": {
                "total_files": len(individual_transcripts),
                "total_words": total_words,
                "unique_words": len(word_counts),
                "avg_words_per_file": sum(t["word_count"] for t in individual_transcripts) / len(individual_transcripts)
            },
            "cache_info": {
                "cached_count": cached_count,
                "missing_count": missing_count,
                "cache_hit_rate": cached_count / len(filenames) if filenames else 0
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in batch whisper analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


@router.post("/inferences/whisper-accuracy")
async def get_whisper_accuracy(request: Request):
    """
    Get whisper prediction from cache and compare with ground truth
    """
    try:
        body = await request.json()
        model = body.get("model", "whisper-base")
        dataset = body.get("dataset")
        dataset_file = body.get("dataset_file")
        file_path = body.get("file_path")
        
        if not dataset_file and not file_path:
            raise HTTPException(status_code=400, detail="Either dataset_file or file_path must be provided")
        
        session_id = get_session_id(request)
        
        # Get file path
        if file_path:
            resolved_path = Path(file_path)
        else:
            resolved_path = resolve_file(dataset, dataset_file, session_id)
        
        if not resolved_path.exists():
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        # Create cache key and get cached prediction
        file_content_hash = hashlib.md5(str(resolved_path).encode()).hexdigest()
        cache_key = f"{model}_{file_content_hash}"
        
        cached_result = await get_result(model, cache_key)
        
        if cached_result is None:
            # If not cached, run inference first
            print(f"DEBUG: No cached result found, running inference for {dataset_file}")
            try:
                # Run inference to get the prediction and cache it
                if dataset and dataset_file:
                    inference_result = await run_inference(model, None, dataset, dataset_file, session_id)
                else:
                    inference_result = await run_inference(model, str(resolved_path), None, None, session_id)
                
                # Now try to get the cached result again
                cached_result = await get_result(model, cache_key)
                if cached_result is None:
                    # If still not cached, use the inference result directly
                    cached_result = {"prediction": inference_result}
            except Exception as e:
                print(f"DEBUG: Failed to run inference: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to run inference for {dataset_file}: {str(e)}")
        
        # Extract transcript from cached result
        if isinstance(cached_result, dict):
            predicted_transcript = cached_result.get("prediction", cached_result.get("transcript", ""))
        else:
            predicted_transcript = str(cached_result)
        
        # Get ground truth from dataset metadata
        ground_truth = ""
        if dataset and dataset_file:
            # Load dataset metadata
            if dataset == "common-voice":
                metadata_path = DATA_DIR / "common_voice_valid_dev" / "common_voice_valid_data_metadata.csv"
            elif dataset == "ravdess":
                metadata_path = DATA_DIR / "ravdess_subset" / "ravdess_subset_metadata.csv"
            elif dataset.startswith("custom:"):
                # For custom datasets, ground truth is not available - skip ground truth extraction
                ground_truth = ""
                metadata_path = None
            else:
                raise HTTPException(status_code=400, detail=f"Unknown dataset: {dataset}")
            
            if metadata_path.exists():
                df = pd.read_csv(metadata_path)
                # Find the row for this file
                # Try both with and without path prefix
                file_rows = df[df['filename'] == dataset_file]
                if file_rows.empty:
                    # Try with path prefix for common-voice
                    if dataset == "common-voice":
                        file_rows = df[df['filename'] == f"cv-valid-dev/{dataset_file}"]
                
                if not file_rows.empty:
                    # Try different column names for transcript/text
                    if dataset == "common-voice":
                        # For common-voice, use 'text' column
                        for col in ['text', 'transcript', 'sentence', 'label']:
                            if col in df.columns:
                                ground_truth = str(file_rows.iloc[0][col])
                                break
                    elif dataset == "ravdess":
                        # For RAVDESS, use 'statement' column
                        for col in ['statement', 'text', 'transcript', 'sentence']:
                            if col in df.columns:
                                ground_truth = str(file_rows.iloc[0][col])
                                break
        
        # If ground truth is not available, we'll return None for metrics but still provide the prediction
        has_ground_truth = bool(ground_truth)
        if not ground_truth:
            print(f"DEBUG: No ground truth found for dataset: {dataset}, file: {dataset_file}. Continuing without accuracy metrics.")
        
        # Calculate accuracy metrics
        def calculate_accuracy(predicted, ground_truth):
            # Clean and normalize both strings
            def clean_text(text):
                # Convert to lowercase
                text = text.lower()
                # Remove punctuation
                text = text.translate(str.maketrans('', '', string.punctuation))
                # Remove extra whitespace and normalize
                text = re.sub(r'\s+', ' ', text).strip()
                return text
            
            pred_clean = clean_text(predicted)
            truth_clean = clean_text(ground_truth)
            
            # Split into words for word-based metrics
            pred_words = pred_clean.split()
            truth_words = truth_clean.split()
            
            # Character-based similarity (after cleaning)
            char_similarity = difflib.SequenceMatcher(None, pred_clean, truth_clean).ratio()
            
            # Word-based similarity (after cleaning)
            word_similarity = difflib.SequenceMatcher(None, pred_words, truth_words).ratio()
            
            # Exact match accuracy
            exact_match = 1.0 if pred_clean == truth_clean else 0.0
            
            # Levenshtein distance (character-level)
            def levenshtein_distance(s1, s2):
                if len(s1) < len(s2):
                    return levenshtein_distance(s2, s1)
                if len(s2) == 0:
                    return len(s1)
                
                previous_row = list(range(len(s2) + 1))
                for i, c1 in enumerate(s1):
                    current_row = [i + 1]
                    for j, c2 in enumerate(s2):
                        insertions = previous_row[j + 1] + 1
                        deletions = current_row[j] + 1
                        substitutions = previous_row[j] + (c1 != c2)
                        current_row.append(min(insertions, deletions, substitutions))
                    previous_row = current_row
                
                return previous_row[-1]
            
            lev_dist = levenshtein_distance(pred_clean, truth_clean)
            
            # Word Error Rate (WER) - standard ASR metric
            def calculate_wer(predicted_words, reference_words):
                # This is a simplified WER calculation using edit distance on word level
                if len(reference_words) == 0:
                    return 1.0 if len(predicted_words) > 0 else 0.0
                
                # Calculate word-level edit distance
                word_lev_dist = levenshtein_distance(predicted_words, reference_words)
                wer = word_lev_dist / len(reference_words)
                return min(wer, 1.0)  # Cap at 1.0
            
            # Character Error Rate (CER)
            def calculate_cer(predicted_chars, reference_chars):
                if len(reference_chars) == 0:
                    return 1.0 if len(predicted_chars) > 0 else 0.0
                
                char_lev_dist = levenshtein_distance(predicted_chars, reference_chars)
                cer = char_lev_dist / len(reference_chars)
                return min(cer, 1.0)  # Cap at 1.0
            
            wer = calculate_wer(pred_words, truth_words)
            cer = calculate_cer(pred_clean, truth_clean)
            
            # Overall accuracy based on word similarity (most intuitive)
            accuracy_percentage = word_similarity * 100
            
            return {
                "accuracy_percentage": accuracy_percentage,
                "word_error_rate": wer,
                "character_error_rate": cer,
                "levenshtein_distance": lev_dist,
                "exact_match": exact_match,
                "character_similarity": char_similarity * 100,
                "word_count_predicted": len(pred_words),
                "word_count_truth": len(truth_words)
            }
        
        # Calculate accuracy metrics only if ground truth is available
        if has_ground_truth:
            accuracy_metrics = calculate_accuracy(predicted_transcript, ground_truth)
            return {
                "predicted_transcript": predicted_transcript,
                "ground_truth": ground_truth,
                **accuracy_metrics
            }
        else:
            # Return just the prediction without ground truth metrics
            return {
                "predicted_transcript": predicted_transcript,
                "ground_truth": "",
                "accuracy_percentage": None,
                "word_error_rate": None,
                "character_error_rate": None,
                "levenshtein_distance": None,
                "exact_match": None,
                "character_similarity": None,
                "word_count_predicted": len(predicted_transcript.split()) if predicted_transcript else 0,
                "word_count_truth": 0
            }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in whisper accuracy calculation: {e}")
        raise HTTPException(status_code=500, detail=f"Accuracy calculation failed: {str(e)}")


@router.post("/inferences/wav2vec2-batch")
async def batch_wav2vec2_prediction(request: Request):
    """
    Get batch wav2vec2 predictions for multiple files and calculate aggregated probabilities.
    Uses cached results when available, only runs model for uncached files.
    """
    try:
        body = await request.json()
        filenames = body.get("filenames", [])
        dataset = body.get("dataset")
        
        if not filenames:
            raise HTTPException(status_code=400, detail="No filenames provided")
        
        if len(filenames) > 50:  # Limit batch size
            raise HTTPException(status_code=400, detail="Too many files. Maximum 50 files per batch.")
        
        session_id = get_session_id(request)
        
        # Process each file
        individual_predictions = []
        predicted_emotions = []  # Store just the predicted emotions for distribution
        cache_stats = {"hits": 0, "misses": 0}
        
        for filename in filenames:
            try:
                # Resolve file path
                if dataset:
                    file_path = resolve_file(dataset, filename, session_id)
                else:
                    file_path = UPLOAD_DIR / filename
                    if not file_path.exists():
                        print(f"Warning: File not found: {file_path}")
                        continue
                
                # Create cache key
                file_content_hash = hashlib.md5(str(file_path).encode()).hexdigest()
                cache_key = f"wav2vec2_detailed_{file_content_hash}"
                
                # Check cache first
                cached_result = await get_result("wav2vec2", cache_key)
                if cached_result is not None:
                    # Use cached result
                    result = cached_result.get("prediction", cached_result)
                    cache_stats["hits"] += 1
                    logger.debug(f"Using cached wav2vec2 result for {filename}")
                else:
                    # Run model and cache result
                    result = await asyncio.to_thread(predict_emotion_wave2vec, str(file_path))
                    await cache_result("wav2vec2", cache_key, {"prediction": result}, ttl=6*60*60)
                    cache_stats["misses"] += 1
                    logger.debug(f"Generated and cached wav2vec2 result for {filename}")
                
                individual_predictions.append({
                    "filename": filename,
                    "predicted_emotion": result["predicted_emotion"],
                    "probabilities": result["probabilities"], 
                    "confidence": result["confidence"]
                })
                
                # Store the predicted emotion for distribution calculation
                predicted_emotions.append(result["predicted_emotion"])
                    
            except Exception as file_error:
                print(f"Error processing {filename}: {file_error}")
                continue
        
        if not individual_predictions:
            raise HTTPException(status_code=404, detail="No valid files could be processed")
        
        # Calculate emotion distribution (percentage of files predicted as each emotion)
        emotion_counts = {}
        for emotion in predicted_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        total_files = len(predicted_emotions)
        emotion_distribution = {}
        for emotion, count in emotion_counts.items():
            emotion_distribution[emotion] = count / total_files
        
        # Find dominant emotion (most frequent prediction)
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])
        
        return {
            "emotion_distribution": emotion_distribution,  # Percentage of files predicted as each emotion
            "emotion_counts": emotion_counts,  # Raw counts
            "individual_predictions": individual_predictions,
            "summary": {
                "total_files": total_files,
                "dominant_emotion": dominant_emotion[0],
                "dominant_count": dominant_emotion[1],
                "dominant_percentage": dominant_emotion[1] / total_files
            },
            "cache_info": {
                "cached_count": cache_stats["hits"],
                "missing_count": cache_stats["misses"],
                "cache_hit_rate": cache_stats["hits"] / (cache_stats["hits"] + cache_stats["misses"]) if (cache_stats["hits"] + cache_stats["misses"]) > 0 else 0
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in batch wav2vec2 prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.post("/inferences/wav2vec2-detailed")
async def get_wav2vec2_detailed_prediction(
    http_request: Request,
    request: dict = Body(..., example={
        "file_path": "/path/to/audio.wav",
        "dataset": "common-voice", 
        "dataset_file": "sample-001.mp3",
        "include_attention": True
    })
):
    """Get detailed wav2vec2 prediction with probabilities for all emotions and ground truth if available"""
    file_path = request.get("file_path")
    dataset = request.get("dataset")
    dataset_file = request.get("dataset_file")
    include_attention = request.get("include_attention", True)  # Default to True for attention extraction
    
    session_id = get_session_id(http_request)
    
    # Resolve file path
    resolved_path = None
    if file_path:
        resolved_path = Path(file_path)
    elif dataset and dataset_file:
        try:
            resolved_path = resolve_file(dataset, dataset_file, session_id)
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(status_code=404, detail=str(e))
    else:
        raise HTTPException(
            status_code=400,
            detail="Missing audio reference. Provide either 'file_path' or 'dataset' + 'dataset_file'."
        )
    
    if not resolved_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {resolved_path}")
    
    # Create cache key for detailed predictions (v3 after fixing attention extraction)
    file_content_hash = hashlib.md5(str(resolved_path).encode()).hexdigest()
    cache_key = f"wav2vec2_detailed_attention_v3_{file_content_hash}"
    
    # Check if result is cached
    cached_result = await get_result("wav2vec2", cache_key)
    if cached_result is not None:
        logger.info(f"Found cached detailed wav2vec2 result for {resolved_path}")
        cached_prediction = cached_result.get("prediction", cached_result)
        
        # Check if cached result has attention data
        cached_has_attention = cached_prediction.get("attention") is not None if isinstance(cached_prediction, dict) else False
        
        # If attention is requested but not in cache, we need to regenerate
        if include_attention and not cached_has_attention:
            logger.info(f"Attention requested but not in cache, regenerating for {resolved_path}")
        else:
            # Debug: Check if cached result has attention
            cached_layers = len(cached_prediction.get("attention", [])) if cached_has_attention else 0
            logger.info(f"Returning cached wav2vec2 result - has attention: {cached_has_attention}, layers: {cached_layers}")
            return cached_prediction
    
    # Get detailed prediction with probabilities and conditionally include attention
    try:
        if include_attention:
            # Use the more expensive attention-enabled function
            detailed_result = await asyncio.to_thread(predict_emotion_wave2vec_with_attention, str(resolved_path))
        else:
            # Use the regular prediction function which is faster
            detailed_result = await asyncio.to_thread(predict_emotion_wave2vec, str(resolved_path))
            # Ensure the result has the expected structure
            if "attention" not in detailed_result:
                detailed_result["attention"] = None
        
        # Try to get ground truth emotion if we have dataset information
        ground_truth_emotion = ""
        if dataset and dataset_file:
            # Only try to get emotion ground truth from RAVDESS dataset
            if dataset == "ravdess":
                metadata_path = DATA_DIR / "ravdess_subset" / "ravdess_subset_metadata.csv"
                if metadata_path.exists():
                    df = pd.read_csv(metadata_path)
                    file_rows = df[df['filename'] == dataset_file]
                    if not file_rows.empty:
                        # Try different column names for emotion
                        for col in ['emotion', 'label', 'category']:
                            if col in df.columns:
                                ground_truth_emotion = str(file_rows.iloc[0][col])
                                break
        
        # Add ground truth to the result if available
        if ground_truth_emotion:
            detailed_result["ground_truth_emotion"] = ground_truth_emotion
        else:
            detailed_result["ground_truth_emotion"] = None
        
        # Cache the detailed result without attention data to avoid memory issues
        cache_data = detailed_result.copy()
        if "attention" in cache_data:
            # Remove attention data from cache to prevent MemoryError
            cache_data["attention"] = None
            logger.info(f"Excluded attention data from cache to prevent memory issues")
        
        await cache_result("wav2vec2", cache_key, {"prediction": cache_data}, ttl=6*60*60)
        logger.info(f"Cached detailed wav2vec2 prediction for {resolved_path} (without attention data)")
        
        # Debug: Log if attention data is present
        has_attention = detailed_result.get("attention") is not None
        logger.info(f"Wav2Vec2 result has attention data: {has_attention}")
        if has_attention:
            attention_shape = f"layers: {len(detailed_result['attention'])}"
            logger.info(f"Attention shape: {attention_shape}")
        
        return detailed_result
        
    except Exception as e:
        logger.error(f"Error getting detailed wav2vec2 prediction: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Return a fallback response instead of raising 500 error
        fallback_result = {
            "predicted_emotion": "unknown",
            "probabilities": {"unknown": 1.0},
            "confidence": 0.0,
            "attention": None,
            "error": str(e),
            "fallback": True
        }
        logger.info("Returning fallback result for wav2vec2 prediction")
        return fallback_result


@router.post("/inferences/embeddings")
async def extract_embeddings_endpoint(
    http_request: Request,
    request: dict = Body(..., example={
        "model": "whisper-base",
        "dataset": "common-voice",
        "files": ["sample-001.mp3", "sample-002.mp3"],
        "reduction_method": "pca",
        "n_components": 3
    })
):
    """Extract embeddings from multiple audio files and optionally reduce dimensions"""
    model = request.get("model")
    dataset = request.get("dataset")
    files = request.get("files", [])
    reduction_method = request.get("reduction_method", "pca")
    n_components = request.get("n_components", 3)
    
    if not model or not dataset or not files:
        raise HTTPException(status_code=400, detail="Model, dataset, and files are required")
    
    session_id = get_session_id(http_request)
    logger.info(f"Extracting embeddings for {len(files)} files with model {model}")
    
    embeddings_data = []
    embeddings_list = []
    
    for filename in files:
        try:
            # Resolve the file path
            resolved_path = resolve_file(dataset, filename, session_id)
            
            # Create cache key for embeddings
            file_content_hash = hashlib.md5(str(resolved_path).encode()).hexdigest()
            cache_key = f"{model}_embeddings_{file_content_hash}"
            
            # Check if embeddings are cached
            cached_embeddings = await get_result(model, cache_key)
            
            if cached_embeddings is not None:
                embedding = cached_embeddings.get("embedding")
                logger.info(f"Using cached embeddings for {filename}")
            else:
                # Extract embeddings based on model type
                if model.startswith("whisper"):
                    model_size = "base" if "base" in model else "large"
                    embedding = await asyncio.to_thread(extract_whisper_embeddings, str(resolved_path), model_size)
                elif model == "wav2vec2":
                    embedding = await asyncio.to_thread(extract_wav2vec2_embeddings, str(resolved_path))
                else:
                    raise HTTPException(status_code=400, detail=f"Embedding extraction not supported for model: {model}")
                
                # Cache the embeddings (24 hours TTL since embeddings don't change)
                await cache_result(model, cache_key, {"embedding": embedding.tolist()}, ttl=24*60*60)
                logger.info(f"Cached embeddings for {filename}")
            
            # Convert back to numpy array if it was cached as list
            if isinstance(embedding, list):
                embedding = np.array(embedding)
            
            embeddings_data.append({
                "filename": filename,
                "embedding": embedding.tolist(),
                "embedding_dim": len(embedding)
            })
            embeddings_list.append(embedding)
            
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Skipping {filename}: {str(e)}")
            continue
        except Exception as e:
            logger.error(f"Error extracting embeddings for {filename}: {str(e)}")
            continue
    
    if not embeddings_list:
        raise HTTPException(status_code=400, detail="No valid embeddings could be extracted")
    
    logger.info(f"Successfully extracted embeddings for {len(embeddings_list)} files")
    
    # Perform dimensionality reduction if requested
    reduced_embeddings = None
    if reduction_method and len(embeddings_list) > 1:
        try:
            reduced_embeddings = await asyncio.to_thread(
                reduce_dimensions, embeddings_list, reduction_method, n_components
            )
            logger.info(f"Reduced {len(embeddings_list)} embeddings from {embeddings_list[0].shape[0]}D to {n_components}D using {reduction_method}")
        except Exception as e:
            logger.warning(f"Dimensionality reduction failed: {str(e)}")
            # Return error details for debugging
            response_error = {
                "error": f"Dimensionality reduction failed: {str(e)}",
                "embeddings_count": len(embeddings_list),
                "embedding_dimension": embeddings_list[0].shape[0] if embeddings_list else 0
            }
            raise HTTPException(status_code=500, detail=response_error)
    
    # Prepare response
    response = {
        "model": model,
        "dataset": dataset,
        "reduction_method": reduction_method,
        "n_components": n_components,
        "embeddings": embeddings_data,
        "total_files": len(embeddings_data),
        "original_dimension": embeddings_list[0].shape[0] if embeddings_list else 0
    }
    
    if reduced_embeddings is not None:
        response["reduced_embeddings"] = [
            {
                "filename": embeddings_data[i]["filename"],
                "coordinates": reduced_embeddings[i].tolist()
            }
            for i in range(len(reduced_embeddings))
        ]
    
    return response


@router.post("/inferences/embeddings/single")
async def extract_single_embedding_endpoint(
    http_request: Request,
    request: dict = Body(..., example={
        "model": "whisper-base",
        "file_path": "/path/to/audio.wav",
        "dataset": "common-voice",
        "dataset_file": "sample-001.mp3"
    })
):
    """Extract embeddings from a single audio file"""
    model = request.get("model")
    file_path = request.get("file_path")
    dataset = request.get("dataset")
    dataset_file = request.get("dataset_file")
    
    if not model:
        raise HTTPException(status_code=400, detail="Model is required")
    
    session_id = get_session_id(http_request)
    
    # Resolve file path
    resolved_path = None
    if file_path:
        resolved_path = Path(file_path)
    elif dataset and dataset_file:
        try:
            resolved_path = resolve_file(dataset, dataset_file, session_id)
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(status_code=404, detail=str(e))
    else:
        raise HTTPException(
            status_code=400,
            detail="Missing audio reference. Provide either 'file_path' or 'dataset' + 'dataset_file'."
        )
    
    if not resolved_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {resolved_path}")
    
    # Create cache key for embeddings
    file_content_hash = hashlib.md5(str(resolved_path).encode()).hexdigest()
    cache_key = f"{model}_embeddings_{file_content_hash}"
    
    # Check if embeddings are cached
    cached_embeddings = await get_result(model, cache_key)
    
    if cached_embeddings is not None:
        embedding = cached_embeddings.get("embedding")
        logger.info(f"Using cached embeddings for {resolved_path}")
    else:
        # Extract embeddings based on model type
        if model.startswith("whisper"):
            model_size = "base" if "base" in model else "large"
            embedding = await asyncio.to_thread(extract_whisper_embeddings, str(resolved_path), model_size)
        elif model == "wav2vec2":
            embedding = await asyncio.to_thread(extract_wav2vec2_embeddings, str(resolved_path))
        else:
            raise HTTPException(status_code=400, detail=f"Embedding extraction not supported for model: {model}")
        
        # Cache the embeddings (24 hours TTL)
        await cache_result(model, cache_key, {"embedding": embedding.tolist()}, ttl=24*60*60)
        logger.info(f"Cached embeddings for {resolved_path}")
    
    # Convert back to numpy array if it was cached as list
    if isinstance(embedding, list):
        embedding = np.array(embedding)
    
    return {
        "model": model,
        "file_path": str(resolved_path),
        "embedding": embedding.tolist(),
        "embedding_dim": len(embedding)
    }
    
@router.post("/inferences/audio-frequency-batch")
async def batch_audio_frequency_analysis(request: Request):
    """
    Extract frequency-domain audio features for multiple files for analysis.
    This provides detailed spectral analysis for both whisper and wav2vec2 models.
    """
    try:
        body = await request.json()
        filenames = body.get("filenames", [])
        dataset = body.get("dataset")
        model = body.get("model", "whisper-base")  # Track which model context this is for
        
        if not filenames:
            raise HTTPException(status_code=400, detail="No filenames provided")
        
        if len(filenames) > 50:  # Limit batch size
            raise HTTPException(status_code=400, detail="Too many files. Maximum 50 files per batch.")
        
        session_id = get_session_id(request)
        logger.info(f"Session ID: {session_id}")
        
        # Process each file
        individual_analyses = []
        all_features = []
        cache_stats = {"hits": 0, "misses": 0}
        
        for filename in filenames:
            try:
                logger.info(f"Processing file: {filename}")
                # Resolve file path
                if dataset:
                    file_path = resolve_file(dataset, filename, session_id)
                    logger.info(f"Resolved file path: {file_path}")
                else:
                    file_path = UPLOAD_DIR / filename
                    if not file_path.exists():
                        logger.warning(f"File not found: {file_path}")
                        continue
                    logger.info(f"Using upload file path: {file_path}")
                
                # Create cache key for audio frequency features
                file_content_hash = hashlib.md5(str(file_path).encode()).hexdigest()
                cache_key = f"audio_frequency_{file_content_hash}"
                
                # Check cache first
                cached_result = await get_result("audio_frequency", cache_key)
                if cached_result is not None:
                    # Use cached result
                    features = cached_result.get("features", cached_result)
                    cache_stats["hits"] += 1
                    logger.debug(f"Using cached audio frequency features for {filename}")
                else:
                    # Extract features and cache result
                    features = await asyncio.to_thread(extract_audio_frequency_features, str(file_path))
                    await cache_result("audio_frequency", cache_key, {"features": features}, ttl=24*60*60)  # 24h cache
                    cache_stats["misses"] += 1
                    logger.debug(f"Generated and cached audio frequency features for {filename}")
                
                individual_analyses.append({
                    "filename": filename,
                    "features": features
                })
                
                # Collect features for aggregate analysis
                all_features.append(features)
                    
            except Exception as file_error:
                print(f"Error processing {filename}: {file_error}")
                continue
        
        if not individual_analyses:
            raise HTTPException(status_code=404, detail="No valid files could be processed for frequency analysis")
        
        # Calculate aggregate statistics across all files
        feature_keys = set()
        for features in all_features:
            feature_keys.update(features.keys())
        
        aggregate_stats = {}
        feature_distributions = {}
        
        for key in feature_keys:
            values = [features.get(key, 0) for features in all_features if key in features]
            if values:
                aggregate_stats[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values))
                }
                
                # Create distribution bins for visualization
                hist, bins = np.histogram(values, bins=10)
                feature_distributions[key] = {
                    "histogram": hist.tolist(),
                    "bins": bins.tolist()
                }
        
        # Identify most common/prevalent features (highest normalized mean values)
        common_features = []
        for key, stats in aggregate_stats.items():
            # Normalize mean by the range to get a comparable score
            feature_range = stats["max"] - stats["min"]
            if feature_range > 0:
                normalized_mean = (stats["mean"] - stats["min"]) / feature_range
                
                # Calculate stability (inverse of coefficient of variation)
                stability = 1.0
                if stats["mean"] != 0:
                    cv = stats["std"] / abs(stats["mean"])
                    stability = 1.0 / (1.0 + cv)  # Higher stability = lower variation
                
                common_features.append({
                    "feature": key,
                    "normalized_mean": float(normalized_mean),
                    "stability_score": float(stability),
                    "prevalence_score": float(normalized_mean * stability),  # Combined score
                    "mean": stats["mean"],
                    "std": stats["std"]
                })
        
        # Sort by prevalence score (descending) - features that are both high and stable
        common_features.sort(key=lambda x: x["prevalence_score"], reverse=True)
        
        # Categorize features for better presentation
        feature_categories = {
            "spectral": [f for f in feature_keys if f.startswith(("spectral_", "zero_crossing"))],
            "energy": [f for f in feature_keys if "rms" in f or "energy" in f],
            "mfcc": [f for f in feature_keys if f.startswith("mfcc_")],
            "chroma": [f for f in feature_keys if f.startswith("chroma_")],
            "tonnetz": [f for f in feature_keys if f.startswith("tonnetz_")],
            "temporal": [f for f in feature_keys if f in ["tempo", "duration"]],
            "metadata": [f for f in feature_keys if f in ["sample_rate"]]
        }
        
        return {
            "model_context": model,
            "individual_analyses": individual_analyses,
            "aggregate_statistics": aggregate_stats,
            "feature_distributions": feature_distributions,
            "most_common_features": common_features[:10],  # Top 10 most common/prevalent
            "feature_categories": feature_categories,
            "summary": {
                "total_files": len(individual_analyses),
                "total_features_extracted": len(feature_keys),
                "avg_duration": aggregate_stats.get("duration", {}).get("mean", 0),
                "avg_tempo": aggregate_stats.get("tempo", {}).get("mean", 0)
            },
            "cache_info": {
                "cached_count": cache_stats["hits"],
                "missing_count": cache_stats["misses"],
                "cache_hit_rate": cache_stats["hits"] / (cache_stats["hits"] + cache_stats["misses"]) if (cache_stats["hits"] + cache_stats["misses"]) > 0 else 0
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in batch audio frequency analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Audio frequency analysis failed: {str(e)}")


@router.post("/inferences/whisper-attention")
async def get_whisper_with_attention(
    http_request: Request,
    request: dict = Body(..., example={
        "model": "whisper-base",
        "file_path": "/path/to/audio.wav",
        "dataset": "common-voice", 
        "dataset_file": "sample-001.mp3"
    })
):
    """Get Whisper transcription with attention weights"""
    model = request.get("model", "whisper-base")
    file_path = request.get("file_path")
    dataset = request.get("dataset")
    dataset_file = request.get("dataset_file")
    
    session_id = get_session_id(http_request)
    
    # Validate model
    if not model.startswith("whisper"):
        raise HTTPException(status_code=400, detail="This endpoint is only for Whisper models")
    
    # Resolve file path
    resolved_path = None
    if file_path:
        resolved_path = Path(file_path)
    elif dataset and dataset_file:
        try:
            resolved_path = resolve_file(dataset, dataset_file, session_id)
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(status_code=404, detail=str(e))
    else:
        raise HTTPException(
            status_code=400,
            detail="Missing audio reference. Provide either 'file_path' or 'dataset' + 'dataset_file'."
        )
    
    if not resolved_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {resolved_path}")
    
    # Create cache key for attention predictions (v2 to avoid old cache)
    file_content_hash = hashlib.md5(str(resolved_path).encode()).hexdigest()
    cache_key = f"{model}_attention_v2_{file_content_hash}"
    
    # Check if result is cached
    cached_result = await get_result(model, cache_key)
    if cached_result is not None:
        logger.info(f"Returning cached {model} attention result for {resolved_path}")
        cached_prediction = cached_result.get("prediction", cached_result)
        # Debug: Check if cached result has attention
        cached_has_attention = cached_prediction.get("attention") is not None if isinstance(cached_prediction, dict) else False
        cached_layers = len(cached_prediction.get("attention", [])) if cached_has_attention else 0
        logger.info(f"Cached result has attention: {cached_has_attention}, layers: {cached_layers}")
        return cached_prediction
    
    # Get transcription with attention
    try:
        model_size = "base" if "base" in model else "large"
        result = await asyncio.to_thread(transcribe_whisper_with_attention, str(resolved_path), model_size)
        
        # Cache the result
        await cache_result(model, cache_key, {"prediction": result}, ttl=6*60*60)
        logger.info(f"Cached {model} attention prediction for {resolved_path}")
        
        # Debug: Log if attention data is present
        has_attention = result.get("attention") is not None
        logger.info(f"Whisper result has attention data: {has_attention}")
        if has_attention:
            attention_shape = f"layers: {len(result['attention'])}"
            logger.info(f"Attention shape: {attention_shape}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting {model} transcription with attention: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription with attention failed: {str(e)}")


@router.get("/inferences/attention-pairs-test")
async def test_attention_pairs():
    """Test endpoint to verify attention pairs routing works"""
    return {"status": "Attention pairs endpoint is accessible", "message": "Server is working"}

@router.post("/inferences/attention-pairs")
async def extract_attention_pairs_endpoint(
    http_request: Request,
    request: dict = Body(..., example={
        "model": "whisper-base",
        "file_path": "/path/to/audio.wav", 
        "dataset": "common-voice",
        "dataset_file": "sample-001.mp3",
        "layer": 6,
        "head": 0
    })
):
    """Extract word-to-word attention relationships and timestamp-level attention from Whisper model"""
    logger.info("=== ATTENTION PAIRS ENDPOINT START ===")
    try:
        logger.info("=== INSIDE TRY BLOCK ===")
        logger.info(f"Request body: {request}")
        
        # Get session ID following the exact pattern as other endpoints
        session_id = get_session_id(http_request)
        logger.info(f"Extracted session_id: {session_id}")
        
        # Extract parameters
        model = request.get("model", "whisper-base")
        file_path = request.get("file_path")
        dataset = request.get("dataset")
        dataset_file = request.get("dataset_file")
        layer_idx = request.get("layer", 6)
        head_idx = request.get("head", 0)

        # Validate model (following your pattern)
        if "whisper" not in model.lower():
            raise HTTPException(status_code=400, detail="Attention pairs extraction only supports Whisper models")
        
        # Resolve file path following your exact pattern
        resolved_path: Optional[Path] = None
        
        if file_path:
            resolved_path = Path(file_path)
        elif dataset and dataset_file:
            try:
                print(f">>> RESOLVING FILE: dataset='{dataset}', file='{dataset_file}', session='{session_id}'")
                resolved_path = resolve_file(dataset, dataset_file, session_id)
                print(f">>> RESOLVED TO: {resolved_path}")
            except FileNotFoundError as e:
                print(f">>> RESOLVE FILE NOT FOUND: {e}")
                raise HTTPException(status_code=404, detail=str(e))
            except ValueError as e:
                print(f">>> RESOLVE VALUE ERROR: {e}")
                raise HTTPException(status_code=404, detail=str(e))
            except Exception as e:
                print(f">>> RESOLVE UNEXPECTED ERROR: {e}")
                raise HTTPException(status_code=404, detail=f"File resolution error: {str(e)}")
        else:
            raise HTTPException(
                status_code=400,
                detail="Missing audio reference. Provide either 'file_path' or 'dataset' + 'dataset_file'."
            )
        
        logger.info(f"Final resolved_path before existence check: {resolved_path}")
        logger.info(f"Path exists? {resolved_path.exists() if resolved_path else 'None'}")
        
        if not resolved_path.exists():
            logger.error(f"Attention pairs: Audio file not found at {resolved_path}")
            raise HTTPException(status_code=404, detail=f"Audio file not found: {resolved_path}")
        
        # Additional validation for uploaded files
        if file_path and str(resolved_path).startswith("uploads/"):
            logger.info(f"Processing uploaded file for attention pairs: {resolved_path}")
        elif dataset:
            logger.info(f"Processing dataset file for attention pairs: dataset={dataset}, file={dataset_file}")
        
        # Create cache key following your pattern
        file_content_hash = hashlib.md5(str(resolved_path).encode()).hexdigest()
        cache_key = f"{model}_attention_pairs_{file_content_hash}_l{layer_idx}_h{head_idx}"
        
        # Check cache following your pattern
        cached_result = await get_result(model, cache_key)
        if cached_result is not None:
            logger.info(f"Returning cached attention pairs for {resolved_path}")
            return cached_result
        
        # Extract attention pairs using your existing infrastructure
        model_size = "base" if "base" in model else "large"
        
        # Use your existing attention function as base
        attention_result = transcribe_whisper_with_attention(str(resolved_path), model_size)
        
        # Check if attention extraction failed (returns None for research integrity)
        if attention_result is None:
            logger.warning(f"Attention extraction failed for {resolved_path} - returning error instead of mock data")
            raise HTTPException(
                status_code=422, 
                detail="Attention extraction failed. This model/file combination does not support attention analysis."
            )
        
        logger.info(f"Attention result type: {type(attention_result)}")
        logger.info(f"Attention result keys: {list(attention_result.keys()) if isinstance(attention_result, dict) else 'Not a dict'}")
        
        if not attention_result or "attention" not in attention_result:
            logger.error(f"No attention in result. Available keys: {list(attention_result.keys()) if attention_result else 'None'}")
            raise HTTPException(status_code=422, detail="Attention data not available for this model/file combination")
        
        # Also get timestamps separately since attention result doesn't include them
        timestamp_result = transcribe_whisper_with_timestamps(str(resolved_path), model_size)
        
        # Combine attention and timestamp data
        combined_result = {
            **attention_result,
            "chunks": timestamp_result.get("chunks", []),
            "audio": timestamp_result.get("audio"),
            "sample_rate": timestamp_result.get("sample_rate")
        }
        
        # Process attention data into pairs and timeline format
        from app.services.model_loader_service import process_attention_into_pairs
        
        attention_pairs_data = process_attention_into_pairs(
            combined_result,
            str(resolved_path),
            model_size,
            layer_idx,
            head_idx
        )
        
        # Check if processing also failed
        if attention_pairs_data is None:
            logger.warning(f"Attention processing failed for {resolved_path}")
            raise HTTPException(
                status_code=422, 
                detail="Attention data processing failed. Unable to generate attention pairs for this audio."
            )
        
        # Cache result following your pattern
        await cache_result(model, cache_key, attention_pairs_data, ttl=24*60*60)
        
        logger.info(f"Generated attention pairs: {len(attention_pairs_data.get('attention_pairs', []))} pairs")
        
        return attention_pairs_data
        
    except HTTPException:
        logger.info("=== HTTPException caught and re-raised ===")
        raise
    except Exception as e:
        logger.error(f"=== UNEXPECTED EXCEPTION in attention-pairs: {type(e).__name__}: {e} ===")
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Full traceback:\n{tb}")
        raise HTTPException(status_code=500, detail=f"Attention extraction failed: {type(e).__name__}: {str(e)}")
