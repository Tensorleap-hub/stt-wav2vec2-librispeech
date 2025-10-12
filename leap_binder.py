import numpy.typing as npt
from code_loader.contract.visualizer_classes import LeapText, LeapImage, LeapTextMask, LeapGraph
from tensorflow.python.framework.ops import EagerTensor

from librispeech_clean.packages import install_all_packages

install_all_packages()
import nltk
# nltk.download('punkt')
nltk.download('punkt_tab')
#TODO: can replace nltk.download('punkt_tab') : # nltk.data.path.append('/nfs/nltk_data')  # Adjust based on your system
# nltk.download('punkt', download_dir='/nfs/nltk_data')
import pandas as pd
from typing import List
import numpy as np
from code_loader.contract.enums import LeapDataType
import textstat
from textblob import TextBlob
from librispeech_clean.configuration import config
from code_loader.contract.datasetclasses import PreprocessResponse

from librispeech_clean.gcs_utils import download
from librispeech_clean.metrics import ctc_loss, calculate_error_rate_metrics
from librispeech_clean.utils import pad_gt_numeric_labels
from librispeech_clean.visualizers import display_predicted_transcription, display_gt_transcription, \
    display_mel_spectrogram, \
    display_mel_spectrogram_heatmap, display_waveform, display_waveform_heatmap, vis_alignments_pred
from librispeech_clean.wav2vec_processor import ProcessorSingleton
from librosa.feature import spectral_flatness, spectral_contrast, melspectrogram, mfcc, rms, spectral_centroid, \
    spectral_bandwidth, spectral_rolloff, poly_features, zero_crossing_rate

from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_preprocess, tensorleap_input_encoder, \
    tensorleap_custom_visualizer, tensorleap_gt_encoder, tensorleap_metadata, tensorleap_custom_loss, \
    tensorleap_custom_metric


def merge_records_metadata(df: pd.DataFrame):
    fpaths = config.get_parameter('metadata_file_names')
    fbooks, fchapters, fspeakers = fpaths
    fbooks = download(fbooks)
    fchapters = download(fchapters)
    fspeakers = download(fspeakers)
    df_books = pd.read_csv(fbooks, sep='\s+\|\s*', header=None, engine="python", names=["chapter_id", "values", "comment"])
    df_chaps = pd.read_csv(fchapters, sep='|', header=13, engine="python")
    df_speaks = pd.read_csv(fspeakers, sep='\s+\|\s*', header=11, engine="python").reset_index()
    df_speaks.columns = ["id", "gender", "subset", "minutes", "name"]
    df = df.merge(df_speaks, how="left", left_on="speaker_id", right_on="id", suffixes=('', '_speaks')).drop(["id_speaks"], axis=1)
    df = df.merge(df_chaps, how="left", left_on="chapter_id", right_on=';ID    ', suffixes=('', '_chap'))
    df = df.rename(columns={';ID    ': 'id_chaps', 'READER': 'reader', 'MINUTES': 'reader', ' SUBSET           ': 'subset',
                       ' PROJ.': 'project_chap', 'BOOK ID': 'book_id', ' CH. TITLE ': 'chap_title',
                       ' PROJECT TITLE': 'project_title'})
    return df

# -data processing-
@tensorleap_preprocess()
def get_data_subsets() -> List[PreprocessResponse]:
    responses = []
    for dataset_slice, slice_dict in config.get_parameter('dataset_slices').items():
        if slice_dict['path'] is not None:
            fpath = download(slice_dict['path'])
            data = pd.read_csv(fpath, index_col=0)
            data = merge_records_metadata(data)
            data = data.sample(n=slice_dict['n_samples'], random_state=config.get_parameter('seed'))
            response = PreprocessResponse(length=slice_dict['n_samples'], data=data)
            responses.append(response)
    return responses

@tensorleap_input_encoder('input_encoder')
def get_input_audio(idx: int, data: PreprocessResponse, padded: bool = True) -> np.ndarray:
    data = data.data
    audio_gcs_path = data.iloc[idx]['audio_path']
    fpath = download(audio_gcs_path)
    audio_array = np.load(fpath)[0]
    if not padded:
        return audio_array

    padding = config.get_parameter('max_sequence_length') - audio_array.size
    padded_audio_array = np.pad(audio_array, (0, padding))
    return padded_audio_array

def get_input_audio_no_padding(idx: int, data: PreprocessResponse, padded: bool = False) -> np.ndarray:
    data = data.data
    audio_gcs_path = data.iloc[idx]['audio_path']
    fpath = download(audio_gcs_path)
    audio_array = np.load(fpath)[0]
    if not padded:
        return audio_array

    padding = config.get_parameter('max_sequence_length') - audio_array.size
    padded_audio_array = np.pad(audio_array, (0, padding))
    return padded_audio_array

@tensorleap_gt_encoder('numeric_labels')
def get_gt_transcription(idx: int, data: PreprocessResponse):
    data = data.data
    processor = ProcessorSingleton().get_processor()
    transcription = data.iloc[idx]['text']
    numeric_labels = processor.tokenizer.encode(transcription)
    padded_labels = pad_gt_numeric_labels(numeric_labels)
    return np.array(padded_labels).astype('float32')


from typing import Dict, Union

@tensorleap_metadata('metadata_speech_dict')
def get_metadata_speech_dict(idx: int, data: PreprocessResponse) -> Dict[str, Union[int, float, str]]:
    sample = data.data.iloc[idx]
    audio_array = get_input_audio_no_padding(idx, data)
    sr = config.get_parameter('sampling_rate')

    # Extract features
    features = {
            'spectral_flatness': spectral_flatness(y=audio_array),
            'spectral_contrast': spectral_contrast(y=audio_array),
            'melspectrogram': melspectrogram(y=audio_array, sr=sr),
            'mfcc': mfcc(y=audio_array, sr=sr),
            'rms': rms(y=audio_array),
            'spectral_centroid': spectral_centroid(y=audio_array, sr=sr),
            'spectral_bandwidth': spectral_bandwidth(y=audio_array, sr=sr),
            'spectral_rolloff': spectral_rolloff(y=audio_array, sr=sr),
            'poly_features': poly_features(y=audio_array, sr=sr),
            'zero_crossing_rate': zero_crossing_rate(y=audio_array),
    }

    # Extract statistics for each feature
    statistics = {f'{key}_{stat}': float(getattr(value, stat)()) for key, value in features.items() for stat in
                  ['min', 'max', 'mean', 'std']}

    # Additional metadata
    metadata = {
        'index': int(idx),
        'file_name': str(data.data.index[idx]),
        'speaker_id': int(sample['speaker_id']),
        'chapter_id': int(sample['chapter_id']),
        'signal_mean': float(audio_array.mean()),
        'signal_std': float(audio_array.std()),
    }

    # Combine metadata and feature statistics
    result = {**metadata, **statistics}

    # Round numeric values if needed
    result = {k: round(v, 4) if not isinstance(v, str) else v for k, v in result.items()}

    return result


def average_word_length(word_list):
    total_length = sum(len(word) for word in word_list)
    if len(word_list) > 0:
        return total_length / len(word_list)
    else:
        return 0


def min_word_length(word_list):
    min_word = min(word_list, key=len)
    min_length = len(min_word)
    return min_length


def max_word_length(word_list):
    max_word = max(word_list, key=len)
    max_length = len(max_word)
    return max_length


def char_max(transcription_no_spaces):
    char_count = {}

    for char in transcription_no_spaces:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1

    max_char = max(char_count, key=char_count.get)
    max_count = char_count[max_char]

    min_char = min(char_count, key=char_count.get)
    min_count = char_count[min_char]

    return max_char, max_count, min_char, min_count

@tensorleap_metadata('metadata_text_dict')
def get_metadata_text_dict(idx: int, data: PreprocessResponse) -> Dict[str, Union[int, float, str]]:
    data = data.data
    transcription = data.iloc[idx]['text']
    text = TextBlob(transcription)
    transcription_no_spaces = "".join(transcription.split())

    word_list = text.words
    max_char, max_char_count, min_char, min_char_count = char_max(transcription_no_spaces)
    metadata = {
        'word_count': len(word_list),
        'average_word_length': average_word_length(word_list),
        'min_word_length': min_word_length(word_list),
        'max_word_length': max_word_length(word_list),
        'char_count': len(transcription_no_spaces),
        'max_char': max_char,
        'max_char_count': max_char_count,
        'min_char': min_char,
        'min_char_count': min_char_count

    }

    return metadata

@tensorleap_metadata('metadata_readability_text')
def get_metadata_readability_text(idx: int, data: PreprocessResponse):
    data = data.data
    transcription = data.iloc[idx]['text']
    if transcription[-1] != ".":
        transcription = transcription + "."

    metadata = {
        "syllable_count": textstat.syllable_count(transcription, lang='en_US'),
        "lexicon_count": textstat.lexicon_count(transcription, removepunct=True),
        "sentence_count": textstat.sentence_count(transcription),
        "flesch_reading_ease": textstat.flesch_reading_ease(transcription),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(transcription),
        "gunning_fog": textstat.gunning_fog(transcription),
        "dale_chall_readability_score": textstat.dale_chall_readability_score(transcription),
        "difficult_words": textstat.difficult_words(transcription),
        "linsear_write_formula": textstat.linsear_write_formula(transcription),
        "text_standard": textstat.text_standard(transcription)

    }
    return metadata

@tensorleap_metadata('metadata_records')
def get_records_metadata(idx: int, data: PreprocessResponse):
    metadata_dic = {}
    df = data.data
    metadata_dic["gender"] = df.gender.iloc[idx]
    metadata_dic["minutes"] = df.minutes.iloc[idx]
    metadata_dic["project_chap"] = str(df.project_chap.iloc[idx])
    return metadata_dic

@tensorleap_custom_visualizer('transcription', LeapDataType.Text)
def call_display_predicted_transcription(data: npt.NDArray[np.float32]) -> LeapText:
     logits_new=data.transpose((0, 2, 1))
     return display_predicted_transcription(logits_new)

@tensorleap_custom_visualizer('reference', LeapDataType.Text)
def call_display_gt_transcription(data: npt.NDArray[np.float32]) -> LeapText:
     return display_gt_transcription(data)

@tensorleap_custom_visualizer('vis_alignments_pred', LeapDataType.TextMask)
def call_vis_alignments_pred(prediction: np.ndarray, numeric_labels: np.ndarray) -> LeapTextMask:
     prediction_new=prediction.transpose((0, 2, 1))
     return vis_alignments_pred(prediction_new,numeric_labels)

@tensorleap_custom_visualizer('mel_spectrogram', LeapDataType.Image,display_mel_spectrogram_heatmap)
def call_display_mel_spectrogram(data: npt.NDArray[np.float32]) -> LeapImage:
     return display_mel_spectrogram(data)

@tensorleap_custom_visualizer('waveform', LeapDataType.Graph,display_waveform_heatmap)
def call_display_waveform(data: npt.NDArray[np.float32]) -> LeapGraph:
     return display_waveform(data)



@tensorleap_custom_loss('ctc_loss')
def call_ctc_loss(logits: np.ndarray, numeric_labels: np.ndarray) -> EagerTensor:
     logits_new=logits.transpose((0, 2, 1))
     return ctc_loss(logits_new,numeric_labels)

@tensorleap_custom_metric('error_rate_metrics')
def call_calculate_error_rate_metrics(prediction: np.ndarray, numeric_labels: np.ndarray):
    prediction_new = prediction.transpose((0, 2, 1))
    return calculate_error_rate_metrics(prediction_new, numeric_labels)




# leap_binder.set_preprocess(get_data_subsets)
# leap_binder.set_input(get_input_audio, 'audio_array')
# leap_binder.set_ground_truth(get_gt_transcription, 'numeric_labels')
# leap_binder.add_prediction('characters',
#                            ['<pad>', '<s>', '</s>', '<unk>', '|', 'E', 'T', 'A', 'O', 'N', 'I', 'H', 'S', 'R', 'D', 'L',
#                             'U', 'M', 'W', 'C', 'F', 'G', 'Y', 'P', 'B', 'V', 'K', "'", 'X', 'J', 'Q', 'Z'])

# leap_binder.add_custom_metric(calculate_error_rate_metrics, 'error_rate_metrics')
# leap_binder.add_custom_loss(ctc_loss, 'ctc_loss')

# leap_binder.set_metadata(get_metadata_speech_dict, 'metadata_speech_dict')
# leap_binder.set_metadata(get_metadata_text_dict, 'metadata_text_dict')
# leap_binder.set_metadata(get_metadata_readability_text, 'metadata_readability_text')
# leap_binder.set_metadata(get_records_metadata, 'metadata_records')

# leap_binder.set_visualizer(display_predicted_transcription, name='transcription',
#                            visualizer_type=LeapDataType.Text)
# leap_binder.set_visualizer(display_gt_transcription, name='reference',
#                            visualizer_type=LeapDataType.Text)
# leap_binder.set_visualizer(display_mel_spectrogram, name='mel_spectrogram',
#                            heatmap_visualizer=display_mel_spectrogram_heatmap,
#                            visualizer_type=LeapDataType.Image)
# leap_binder.set_visualizer(display_waveform, name='waveform',
#                            heatmap_visualizer=display_waveform_heatmap,
#                            visualizer_type=LeapDataType.Graph)
#
# leap_binder.set_visualizer(vis_alignments_pred, name="vis_alignments_pred", visualizer_type=LeapDataType.TextMask)
#
