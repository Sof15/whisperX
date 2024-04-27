"""Forced alignment according to:
https://pytorch.org/audio/main/tutorials/forced_alignment_for_multilingual_data_tutorial.html 
"""
from torchaudio.pipelines import MMS_FA as bundle
import torch 
import re
from uroman import uroman
from typing import Iterable, List
from .types import SingleSegment
from .audio import SAMPLE_RATE, load_audio
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from .alignment import PUNKT_ABBREVIATIONS

# dictionary with special language-specific cases that
# require different handling during normalization of 
# romanized text
NORMALIZATION_TOKENS_MAP = {
    'el':{
        'o,ti':'oti'
    }
}
SPECIAL_SYMBOLS = '.,!?;:()[]&'

def normalize_uroman(text, language):
    """Function that normalizes the romanized form of a text

    Args:
        text (str): Romanized text
        language (str): Language code e.g. 'en', 'el', 'fr'.

    Returns:
        str: Normalized romanized text
    """
    text = text.lower()
    text = text.replace("’", "'")
    language_tokens_map = NORMALIZATION_TOKENS_MAP.get(language, {})
    for key, value in language_tokens_map.items():
        text = text.replace(key, value)
    text = re.sub("([^a-z' ])", " ", text)
    text = re.sub(' +', ' ', text)
    return text.strip()


def is_nonalpha_indices(text, language):
    """Function that finds the tokens position that got removed during
    text normalization.

    Args:
        text (str): Initial transcript text (before normalization and romanization)
        language (str): Language code e.g. 'en', 'el', 'fr'.

    Returns:
        tuple: (list of discarded tokens' positions, initial text after redundant spaces removal)
    """
    text = re.sub(' +', ' ', text)
    words = text.strip().split()
    language_tokens_map = NORMALIZATION_TOKENS_MAP.get(language, {})
    return [ii for ii, w in enumerate(words) if not w.strip(SPECIAL_SYMBOLS).isalpha() and not uroman(w.strip(SPECIAL_SYMBOLS)) in language_tokens_map], " ".join(words)

def align(
    transcript: Iterable[SingleSegment],
    audio: str,
    device: str,
    language: str
):

    def compute_alignments(waveform: torch.Tensor, transcript: List[str]):
        with torch.inference_mode():
            emission, _ = model(waveform.to(device))
            token_spans = aligner(emission[0], tokenizer(transcript))
        return emission, token_spans

    model = bundle.get_model()
    model.to(device)

    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()

    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        waveform = torch.from_numpy(audio)
    if len(audio.shape) == 1:
        waveform = waveform.unsqueeze(0)
    
    assert bundle.sample_rate==SAMPLE_RATE, f"whisperX is configured with different sample rate ({SAMPLE_RATE}) than the model's ({bundle.sample_rate})"

    aligned_segments = []

    for segment in transcript:
        text = segment["text"]
        
        text_normalized = normalize_uroman(uroman(text, language=language), language=language)
      
        emission, token_spans = compute_alignments(waveform, text_normalized.split())

        ratio = waveform.size(1) / emission.size(1) / SAMPLE_RATE
        word_segments = []
        
        digit_indices, initial_text_norm = is_nonalpha_indices(text, language=language)
        
        punkt_param = PunktParameters()
        punkt_param.abbrev_types = set(PUNKT_ABBREVIATIONS)
        sentence_splitter = PunktSentenceTokenizer(punkt_param)
        sentence_spans = list(sentence_splitter.span_tokenize(initial_text_norm))

        # add dummy token spans for digits 
        for idx in digit_indices:
            token_spans = token_spans[:idx] + [None] + token_spans[idx:]

        for t_spans, chars in zip(token_spans, initial_text_norm.split()):
            if t_spans is None:
                word_segments.append({"word":chars})    
            else:
                t0, t1 = ratio * t_spans[0].start, ratio * t_spans[-1].end
                word_segments.append({"word":chars, "start":t0, "end":t1})
            
    
        start_idx = 0
     
        for (sstart, send) in sentence_spans:
            sentence_span_text = initial_text_norm[sstart:send]
            sentence_span_first_word_idx, sentence_span_last_word_idx = start_idx, start_idx + len(sentence_span_text.split())-1
            sentence_span_start = word_segments[sentence_span_first_word_idx]
            sentence_span_end = word_segments[sentence_span_last_word_idx]
            start_idx += len(sentence_span_text.split())
            aligned_segments.append({
                "start": sentence_span_start["start"],
                "end": sentence_span_end["end"], 
                "text": sentence_span_text,
                "words": word_segments[sentence_span_first_word_idx:sentence_span_last_word_idx+1]
            })
        word_segments=[]
        for segment in aligned_segments:
            word_segments += segment['words']
    return {"segments": aligned_segments, "word_segments": word_segments}


    
