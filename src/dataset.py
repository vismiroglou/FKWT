import json
import torch
import os
from torchaudio.datasets import SPEECHCOMMANDS
import torchaudio
import numpy as np
import soundfile
from glob import glob

class SpeechCommands(SPEECHCOMMANDS):
    def __init__(self, 
                 root,
                 audio_config: dict,
                 labels_map: str,
                 subset: str,
                 download: bool = True,
                 bg_prob = 0.7,
                 normalize=True,
                 r_min=0.85,
                 r_max=1.15,
                 s_min=-0.1,
                 s_max=0.1,
                 n_time_masks=2,
                 time_mask_width=25,
                 n_freq_masks=2,
                 freq_mask_width=7,
                 augment=False
                 ) -> None:
        super().__init__(root, "speech_commands_v0.02", "SpeechCommands", download, subset)
        '''
        args:
        root (str): Dataset root directory
        audio_config (dict): Audio configuration
        labels_map (str): Path to the labels map json file. 
        subset (str): Train/validation/test subset
        download (bool): Whether to download the dataset.
        bg_prob (float): Probability of adding background noise
        normalize (bool): Whether to normalize the audio
        n_time_masks (int): Number of time masks
        time_mask_width (int): Maximum width of time mask
        n_freq_masks (int): Number of frequency masks
        freq_mask_width (int): Maximum width of frequency mask
        seed (int): Seed for reproducibility         
        '''
        with open(labels_map, 'r') as fd:
                self.labels_map = json.load(fd)

        # Audio Config
        self.sr = audio_config.get('sr', 16000)
        self.n_fft = audio_config.get('n_fft', 400)
        self.win_len = audio_config.get('win_len', 400)
        self.hop_len = audio_config.get('hop_len', 160)
        self.f_min = audio_config.get('f_min', 50)
        self.f_max = audio_config.get('f_max', 8000)
        self.n_mels = audio_config.get('n_mels', 80)
        self.duration = audio_config.get('duration', 1.0)
        self.num_frames = audio_config.get('num_frames', None)

        self.normalize = normalize

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_fft=self.n_fft, win_length=self.win_len, hop_length=self.hop_len,
            f_min=self.f_min, f_max=self.f_max,
            n_mels=self.n_mels, power=2.
        )

        # Augmentation arguments
        self.augment=augment
        if self.augment:
            # Resample
            self.r_max = r_max
            self.r_min = r_min

            # Background
            self.bg_files = glob(os.path.join(root, "SpeechCommands", "speech_commands_v0.02", "_background_noise_", '*.wav'))
            self.bg_prob = bg_prob

            # Time-shift
            self.s_min = s_min
            self.s_max = s_max

            # Masking
            self.n_time_masks = n_time_masks
            self.time_mask_width = time_mask_width
            self.n_freq_masks = n_freq_masks
            self.freq_mask_width = freq_mask_width


    def resample(self, x):
        r_min = self.r_min
        r_max = self.r_max
        sr_new = int(self.sr * np.random.uniform(r_min, r_max))
        x_resampled = torchaudio.transforms.Resample(self.sr, sr_new)(x)
        return x_resampled
    

    def time_shift(self, x):
        sr = self.sr
        s_min = self.s_min
        s_max = self.s_max
        start = int(np.random.uniform(sr * s_min, sr * s_max))
        if start >= 0:
            shifted_x = torch.cat((x[:, start:], torch.FloatTensor(np.random.uniform(-0.001, 0.001, start)).unsqueeze(0)), dim=1)
        else:
            shifted_x = torch.cat((torch.FloatTensor(np.random.uniform(-0.001, 0.001, -start)).unsqueeze(0), x[:, :start]), dim=1)
        return shifted_x
   
    
    def spec_augment(self, mel_spec):
        mel_spec_copy = mel_spec.clone()  # Make a copy of the input mel_spec

        for _ in range(self.n_time_masks):
            offset = np.random.randint(0, self.time_mask_width)
            begin = np.random.randint(0, mel_spec_copy.shape[2] - offset)
            mel_spec_copy[:, :, begin: begin + offset] = 0.0
        
        for _ in range(self.n_freq_masks):
            offset = np.random.randint(0, self.freq_mask_width)
            begin = np.random.randint(0, mel_spec_copy.shape[1] - offset)
            mel_spec_copy[:, begin: begin + offset, :] = 0.0

        return mel_spec_copy


    def add_background(self, x):
        bg_path = np.random.choice(self.bg_files)
        bg_wav = soundfile.read(bg_path)[0]
        index = np.random.randint(0, len(bg_wav-len(x)))
        background_cropped = torch.tensor(0.1*bg_wav[index:index+len(x)])
        wav_with_bg = x.add(background_cropped).float()
        return wav_with_bg


    def __getitem__(self, n: int):
        # Get waveform from speech commands
        items =  super().__getitem__(n)
        audio = items[0]
        sr = items[1]
        
        # Add background
        if self.augment:
            if np.random.random() < self.bg_prob:
                audio = self.add_background(audio)

        # Resample
        # if self.augment:
        #     audio = self.resample(audio)

        # SpeechCommands has a fixed duration of 1.
        min_samples = int(sr * self.duration)
        if audio.shape[1] < min_samples:
            tile_size = (min_samples // audio.shape[1]) + 1
            audio = audio.repeat(1, tile_size)
            audio = audio[:, :min_samples]
    
        label_str = items[2]
        label = self.labels_map[label_str]

        # # Time-shift
        # if self.augment:
        #     audio = self.time_shift(audio)

        # Computing log mel spectrogram
        audio = self.melspec(audio)
        audio = (audio + torch.finfo().eps).log()

        if self.num_frames is not None:
            audio = audio[:, :, :self.num_frames]
    
        if self.normalize:
            mean = torch.mean(audio, [1, 2], keepdims=True)
            std = torch.std(audio, [1, 2], keepdims=True)
            audio = (audio - mean) / (std + 1e-8)

        # Applying SpecAug
        if self.augment:
            audio = self.spec_augment(audio)
        return audio, label
    
    
