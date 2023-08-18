import torch
import wave
# import cv2
# import dlib
import numpy as np


def multimodal_concat(inputs_embeds, audio_feature):
    '''
    Input:  text embedding / audio feature
    Output: multimodal fused embedding
    Used: model.forward(), model.inference()
    '''
    audio_feature = torch.unsqueeze(audio_feature, dim=1)
    # [batch, audio_feature_dim] -> [batch, max_length, audio_feature_dim]
    audio_feature = audio_feature.repeat(1, len(inputs_embeds[0]), 1)
    # [batch, max_length, audio_feature_dim + word_dimension]
    x = torch.cat((inputs_embeds, audio_feature), dim=2)
    return x


def forced_alignment_multimodal_concat(inputs_embeds, audio_feature):
    '''
    Input:  text embedding / audio feature
    Output: multimodal fused embedding
    Used: model.forward(), model.inference()
    '''
    inputs_embeds = torch.unsqueeze(
        inputs_embeds, dim=2)  # [batch, max_length, 1, word_dimension]
    # [batch, max_length, 26, word_dimension]
    x = torch.cat((inputs_embeds, audio_feature), dim=2)
    x = x.view(x.shape[0], x.shape[1], -1)  # [batch, max_length, -1]
    return x


def pad(inputs, padding_size):
    '''
    Used: dataset, model.inference()
    '''
    # tmp = [0 for i in range(max_length)]
    tmp = torch.zeros(padding_size)
    if len(inputs) > padding_size:
        tmp[:len(inputs)] = inputs[:padding_size]  # truncation
    else:
        tmp[:len(inputs)] = inputs  # padding
    return tmp


def get_wav_duration(file_path):
    '''
    Used: dataset, model.inference()
    '''
    with wave.open(file_path, 'rb') as wav_file:
        num_frames = wav_file.getnframes()  # Get the number of frames in the WAV file
        # Get the frame rate (number of frames per second)
        frame_rate = wav_file.getframerate()
        duration = num_frames / frame_rate  # Calculate the duration in seconds

        return duration


def audio_pad(inputs, padding_size):
    '''
    Used: dataset, model.inference()
    '''
    tmp = torch.zeros(padding_size, inputs.shape[-1])
    if inputs.shape[0] > padding_size:
        tmp[:inputs.shape[0], :] = inputs[:padding_size]  # truncation
    else:
        tmp[:inputs.shape[0], :] = inputs  # padding
    return tmp


def audio_word_align(waveform, audio_path, start, end, audio_padding=50):
    '''
    Used: dataset, model.inference()
    '''
    waveform = torch.squeeze(waveform)

    duration = get_wav_duration(audio_path)

    a = (waveform.shape[0] / duration)
    waveform_start = torch.tensor(start).clone() * a
    waveform_start = [int(x)+1 for x in waveform_start]
    waveform_end = torch.tensor(end).clone() * a
    waveform_end = [int(x)+1 for x in waveform_end]

    audio_feature = []
    for i, (s, e) in enumerate(zip(waveform_start, waveform_end)):
        if (i != 0) and (s == 1) and (e == 1):  # padding appear
            word_waveform = torch.zeros(audio_padding, waveform.shape[-1])
        else:
            # split waveform along to word duration
            word_waveform = waveform[s:e, :]
            word_waveform = audio_pad(word_waveform, audio_padding)
        audio_feature.append(word_waveform)
    torch_audio_feature = torch.stack(audio_feature, dim=0)  # list to torch.tensor
    return torch_audio_feature, waveform_start


def get_aligned_landmark(landmark_set, waveform_start):
    output_landmark = []
    for s in waveform_start:
        if s >= landmark_set.shape[0]:  # padding appear
            output_landmark.append(torch.zeros(landmark_set.shape[-1]))
        else:
            output_landmark.append(landmark_set[s])

    return torch.stack(output_landmark, dim=0)  # list to torch.tensor
