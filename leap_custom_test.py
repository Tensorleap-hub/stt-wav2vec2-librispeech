import urllib
from os.path import exists

import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort
import tensorflow as tf
# from onnx2kerastl.customonnxlayer import onnx_custom_objects_map
from onnx2kerastl.customonnxlayer.onnxsqrt import OnnxSqrt
from onnx2kerastl.customonnxlayer.onnxreducemean import OnnxReduceMean
from onnx2kerastl.customonnxlayer.onnxerf import OnnxErf
# from librispeech_clean.custom_layers import *
from leap_binder import get_data_subsets, get_input_audio, get_gt_transcription, get_metadata_speech_dict, \
    get_metadata_text_dict, get_metadata_readability_text
from librispeech_clean.metrics import ctc_loss, calculate_error_rate_metrics
from librispeech_clean.visualizers import display_predicted_transcription, display_gt_transcription, display_mel_spectrogram, \
    display_waveform
from librispeech_clean.wav2vec_processor import ProcessorSingleton

from librosa.feature import rms, melspectrogram, spectral_flatness, spectral_contrast


if __name__ == '__main__':

    onnx_model_path = 'model/wav2vec.onnx'
    keras_model_path = 'model/wav2vec.h5'

    if not exists(onnx_model_path):
        print("Downloading wav2vec ONNX for inference")
        urllib.request.urlretrieve(
            'https://storage.googleapis.com/example-datasets-47ml982d/wav2vec/wav2vec.onnx',
            onnx_model_path)
    keras_model = tf.keras.models.load_model(keras_model_path, custom_objects={'OnnxErf': OnnxErf, 'OnnxReduceMean':
        OnnxReduceMean, 'OnnxSqrt': OnnxSqrt})
    ort_session = ort.InferenceSession(onnx_model_path)
    tokenizer = ProcessorSingleton().get_processor()

    responses = get_data_subsets()
    data = responses[2]
    indices = range(7)
    for idx in indices:
        sample = get_input_audio(idx, data)
        gt = get_gt_transcription(idx, data)
        metadata_readability_text = get_metadata_readability_text(idx, data)
        metadata_speech = get_metadata_speech_dict(idx, data)
        metadata_text = get_metadata_text_dict(idx, data)

        # fig, axes = plt.subplots(2, 1)
        # waveform = display_waveform(sample)
        # axes[0].plot(waveform.data)
        # mel_spectrogram = display_mel_spectrogram(sample)
        # axes[1].imshow(mel_spectrogram.data)

        batched_input = tf.expand_dims(sample, 0)

        batched_gt = tf.expand_dims(gt, 0)

        keras_logits = keras_model(batched_input)
        keras_predicted_ids = np.argmax(keras_logits[0, ...].numpy(), axis=0)

        keras_transcribed_text = tokenizer.batch_decode(keras_predicted_ids)
        keras_logits = keras_logits.numpy().transpose((0, 2, 1))
        #
        loss = ctc_loss(logits=keras_logits, numeric_labels=batched_gt)
        error_rate_metrics = calculate_error_rate_metrics(prediction=keras_logits,
                                                          numeric_labels=batched_gt.numpy())

        predicted_text = display_predicted_transcription(keras_logits[0, ...].numpy())
        gt_text = display_gt_transcription(batched_gt[0, ...].numpy())
    print("finish")

