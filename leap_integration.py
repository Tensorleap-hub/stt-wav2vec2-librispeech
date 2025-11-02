from onnx2kerastl.customonnxlayer.onnxsqrt import OnnxSqrt
from onnx2kerastl.customonnxlayer.onnxreducemean import OnnxReduceMean
from onnx2kerastl.customonnxlayer.onnxerf import OnnxErf
from leap_binder import *
import tensorflow as tf
from code_loader.contract.datasetclasses import PredictionTypeHandler
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_load_model, tensorleap_integration_test

prediction_type1 = PredictionTypeHandler('characters',
                           ['<pad>', '<s>', '</s>', '<unk>', '|', 'E', 'T', 'A', 'O', 'N', 'I', 'H', 'S', 'R', 'D', 'L',
                            'U', 'M', 'W', 'C', 'F', 'G', 'Y', 'P', 'B', 'V', 'K', "'", 'X', 'J', 'Q', 'Z'])


@tensorleap_load_model([prediction_type1])
def load_model():
    keras_model_path = 'model/wav2vec.h5'
    keras_model = tf.keras.models.load_model(keras_model_path, custom_objects={'OnnxErf': OnnxErf, 'OnnxReduceMean':
        OnnxReduceMean, 'OnnxSqrt': OnnxSqrt})
    return keras_model




@tensorleap_integration_test()
def check_custom_integration(idx, data):
    print("started custom tests")
    keras_model = load_model()
    batched_input = get_input_audio(idx, data)
    batched_gt = get_gt_transcription(idx, data)
    keras_logits = keras_model(batched_input)
    # metrics
    loss = call_ctc_loss(logits=keras_logits, numeric_labels=batched_gt)
    error_rate_metrics = call_calculate_error_rate_metrics(prediction=keras_logits,
                                                      numeric_labels=batched_gt)

    # vis
    waveform = call_display_waveform(batched_input)
    mel_spectrogram = call_display_mel_spectrogram(batched_input)
    transcription = call_display_predicted_transcription(keras_logits)
    reference = call_display_gt_transcription(batched_gt)
    alignmentsPred = call_vis_alignments_pred(keras_logits, batched_gt)


    # metadata
    metadata_readability_text = get_metadata_readability_text(idx, data)
    metadata_speech = get_metadata_speech_dict(idx, data)
    metadata_text = get_metadata_text_dict(idx, data)



if __name__ == '__main__':
    idx=0
    responses = get_data_subsets()[2]
    check_custom_integration(idx, responses)
