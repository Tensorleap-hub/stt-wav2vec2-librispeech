import numpy as np
from code_loader.contract.visualizer_classes import LeapText, LeapImage, LeapGraph, LeapTextMask
import numpy.typing as npt

from librispeech_clean.configuration import config
from librispeech_clean.utils import remove_trailing_zeros, normalize_array
from librispeech_clean.wav2vec_processor import ProcessorSingleton
from librosa.feature import melspectrogram, rms
from librosa import power_to_db, resample

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm as cmx
from jiwer import process_characters

cmap = plt.get_cmap('magma')
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)


def display_predicted_transcription(data: npt.NDArray[np.float32]) -> LeapText:
    data = np.squeeze(data)
    processor = ProcessorSingleton().get_processor()
    predicted_ids = np.argmax(data, axis=1)
    text = [processor.decode(predicted_ids)]
    return LeapText(text)


def display_gt_transcription(data: npt.NDArray[np.float32]) -> LeapText:
    data = np.squeeze(data)
    numeric_labels = remove_trailing_zeros(data)
    processor = ProcessorSingleton().get_processor()
    text = [processor.tokenizer.decode(numeric_labels)]
    return LeapText(text)


def display_mel_spectrogram(data: npt.NDArray[np.float32]) -> LeapImage:
    # data_trimmed = remove_trailing_zeros(data)
    data = np.squeeze(data)
    resized_data = data[:config.get_parameter('clip_visualizers')]
    ms = melspectrogram(y=resized_data, sr=config.get_parameter('sampling_rate'), )
    ms_db = power_to_db(ms, ref=np.max)
    scaled_ms = normalize_array(ms_db)
    colored_depth = scalarMap.to_rgba(scaled_ms)[..., :-1]
    res = colored_depth.astype(np.float32) * 255.0
    return LeapImage(res)


def display_mel_spectrogram_heatmap(data: npt.NDArray[np.float32]) -> np.ndarray:
    # data_trimmed = remove_trailing_zeros(data)
    data = np.squeeze(data)
    resized_data = data[:config.get_parameter('clip_visualizers')]
    agg_data = np.tile(rms(y=resized_data), [128, 1])
    res = np.expand_dims(agg_data, -1)
    return res


def display_waveform(data: npt.NDArray[np.float32]) -> LeapGraph:
    sr = config.get_parameter('sampling_rate')
    sr_target = sr // 10
    downsampled_signal = resample(data, orig_sr=sr, target_sr=sr_target)
    resized_data = downsampled_signal[:config.get_parameter('clip_visualizers')]
    # data_trimmed = remove_trailing_zeros(data)
    resized_data = data[:config.get_parameter('clip_visualizers')]
    resized_data = resized_data.reshape(-1, 10).mean(1)
    # down_sampled_data = resample(resized_data,
    #                              orig_sr=config.get_parameter('sampling_rate'),
    #                              target_sr=config.get_parameter('sampling_rate') // 10)
    res = resized_data.reshape(-1, 1)
    return LeapGraph(res)


def display_waveform_heatmap(data: npt.NDArray[np.float32]) -> np.ndarray:
    # data_trimmed = remove_trailing_zeros(data)
    resized_data = data[:config.get_parameter('clip_visualizers')]
    resized_data = resized_data.reshape(-1, 10).mean(1)
    # down_sampled_data = resample(resized_data,
    #                              orig_sr=config.get_parameter('sampling_rate'),
    #                              target_sr=config.get_parameter('sampling_rate') // 10)
    res = resized_data.reshape(-1, 1)
    return res



def vis_alignments_pred(prediction: np.ndarray, numeric_labels: np.ndarray) -> LeapTextMask:
    """
    Returns: LeapTextMask Vis
    mask: npt.NDArray[np.uint8]: length of text with labels in each index
    text: List[str]
    labels: List[str]
    type: LeapDataType = LeapDataType.TextMask
    """
    prediction = np.squeeze(prediction)
    numeric_labels = np.squeeze(numeric_labels)
    numeric_labels = remove_trailing_zeros(numeric_labels)
    processor = ProcessorSingleton().get_processor()
    processed_pred = np.argmax(prediction, 1)
    transcription = processor.decode(processed_pred)
    reference = processor.tokenizer.decode(numeric_labels)

    character_process_out = process_characters(reference, transcription)
    text = list(transcription)
    mask = np.zeros(len(text), dtype=np.uint8)
    labels = ["-", "equal", "insert", "deletion", "substitute", "delete"]

    alignments = character_process_out.alignments
    for align in alignments[0]:
        align_type = align.type
        # start, end = align.ref_start_idx, align.ref_end_idx
        start, end = align.hyp_start_idx, align.hyp_end_idx
        mask[start: end] = labels.index(align_type)

    length = config.get_parameter('text_visualizers')
    if len(text) < length:
        p = (length-len(text))
        text += [""]*p
        mask = np.concatenate([mask, np.zeros(p, dtype=np.uint8)])
    return LeapTextMask(mask, text, labels)



def vis_alignments_heatmap(data: npt.NDArray[np.float32]) -> np.ndarray:
    data = np.squeeze(data)
    length = config.get_parameter('text_visualizers')
    resized_data = np.zeros(length)
    if len(data) > length:
        data = data[:length+1]
    resized_data[0:len(data)] = data
    return resized_data


