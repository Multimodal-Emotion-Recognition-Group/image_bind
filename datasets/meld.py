import os
from pathlib import Path
from PIL import Image
from typing import Optional, Callable
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from moviepy import VideoFileClip
import random
import csv



import lora__impl__.data
from lora__impl__.models.imagebind_model import ModalityType


class MeldDataset(Dataset):
    def __init__(self, csv_path: str, transform: Optional[Callable] = None,
                 split = 'train', arbitrary_size = 1.0, shuffle=False, seed=59, device: str = 'cpu'):

        self.csv_path = csv_path
        self.transform = transform
        self.device = device
        self.seed = seed

        self.classes = sorted(get_unique_classes(csv_path))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}

        self.split_paths = []

        # [text_data, video_path, audio_path, cls]
        self.train_paths = []
        self.dev_paths = []
        self.test_paths = []

        # arbitrary_size allows change target size of split
        if split == 'train':
            vids_path = '/'.join(self.csv_path.split('/')[:-1]) + '/train_splits'
            wavs_path = '/'.join(self.csv_path.split('/')[:-1]) + '/train_wavs'
            self.train_paths = create_dataset_split(self.csv_path, vids_path, wavs_path)
            if shuffle: random.Random(self.seed).shuffle(self.train_paths)
            self.split_paths = self.train_paths[:int(len(self.train_paths)*arbitrary_size)]

        elif split == 'dev':
            vids_path = '/'.join(self.csv_path.split('/')[:-1]) + '/dev_splits'
            wavs_path = '/'.join(self.csv_path.split('/')[:-1]) + '/dev_wavs'
            self.dev_paths = create_dataset_split(self.csv_path, vids_path, wavs_path)
            if shuffle: random.Random(self.seed).shuffle(self.dev_paths)
            self.split_paths = self.dev_paths[:int(len(self.dev_paths)*arbitrary_size)]

        elif split == 'test':
            vids_path = '/'.join(self.csv_path.split('/')[:-1]) + '/test_splits'
            wavs_path = '/'.join(self.csv_path.split('/')[:-1]) + '/test_wavs'
            self.test_paths = create_dataset_split(self.csv_path, vids_path, wavs_path)
            if shuffle: random.Random(self.seed).shuffle(self.test_paths)
            self.split_paths = self.test_paths[:int(len(self.test_paths)*arbitrary_size)]
        else:
            raise ValueError(f"Invalid split argument. Expected 'train' or 'test', got {split}")

    def __getitem__(self, index):
        text, videop, audiop, label = self.split_paths[index]

        text_tkn = lora__impl__.data.load_and_transform_text([text], self.device)
        images = lora__impl__.data.load_and_transform_video_data([videop], self.device)
        mel_spt = lora__impl__.data.load_and_transform_audio_data([audiop], self.device)
        label_tkn = lora__impl__.data.load_and_transform_text([label], self.device)

        return images, ModalityType.VISION, text_tkn, ModalityType.TEXT, mel_spt, ModalityType.AUDIO, label_tkn, ModalityType.TEXT

    def __len__(self):
        return len(self.split_paths)



# some MELD helpers
def get_ds_info(csv_path):
    if csv_path is None:
        csv_path = '../../MELD.Raw/dev_sent_emo.csv'

    df = pd.read_csv(csv_path)
    print(df.columns)

    row_ex = df.iloc[0]
    print(row_ex)

    plt.figure(figsize=(12, 6))
    plt.hist(df['Emotion'], bins='auto')
    plt.show()

    emo_rates = {}
    for ue in df['Emotion'].unique():
        emo_rates[ue] = len(df[df['Emotion'] == ue]) / len(df['Emotion'])

    for k, v in emo_rates.items():
        print(f'{k} rate: {v:.4f}')

def get_unique_classes(csv_path):
    if csv_path is None:
        csv_path = '../../MELD.Raw/dev_sent_emo.csv'

    df = pd.read_csv(csv_path)
    return df['Emotion'].unique()

def get_pil_frames(video_path, num_frames=15):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
        frame_count += 1

    cap.release()

    if len(frames) < num_frames:
        raise ValueError(f"Video contains fewer than {num_frames} frames.")

    return frames

def get_wav_from_mp4(video_path, out_dir):
    out_fname = Path(video_path).stem
    output_path = Path(out_dir + f'/{out_fname}.wav')
    if output_path.exists():
        return output_path

    # if not exists convert mp4 to wav
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_path, codec='pcm_s16le', fps=16000)
    return output_path

def create_dataset_split(csv_path, videos_path, wavs_path):
    df = pd.read_csv(csv_path)
    label_encoder = LabelEncoder()

    # not used
    df['Emotion_encoded'] = label_encoder.fit_transform(df['Emotion'])

    Xy = []
    video_files = [file for file in os.listdir(videos_path)
                  if file.endswith('.mp4') and not file.startswith('.')]

    for video_file in video_files:

        v_path = Path(videos_path + f'/{video_file}')
        try:
            id_info = video_file.split('.mp4')[0].split('_')
            if (len(id_info) != 2
                or not id_info[0][3:].isdigit()
                or not id_info[1][3:].isdigit()):

                raise ValueError(f'invalid video fname format {video_file}')

            diag_id, utt_id = int(id_info[0][3:]), int(id_info[1][3:])
            utt_row = df[(df['Dialogue_ID'] == diag_id) & (df['Utterance_ID'] == utt_id)]

            if not utt_row.empty:
                true_label = utt_row['Emotion'].values[0]
                video_path = v_path
                audio_data_path = get_wav_from_mp4(v_path, wavs_path)
                text_data = utt_row['Utterance'].values[0]

                Xy.append([text_data, video_path, audio_data_path, true_label])
            else:
                print(f'{video_file} has empty row in dataframe')

        except Exception:
            continue
            #print(f'Invalid data for video {v_path}')

    return Xy

if __name__ =='__main__':
    # example of .mp4 that seems fine but can't be read by VideoFileClip
    # video = VideoFileClip('../../MELD.Raw/train/train_splits/dia103_utt5.mp4')
    
    # train_csv_path = '../../MELD.Raw/dev/dev_sent_emo.csv'
    # videos_path = '../../MELD.Raw/dev/dev_splits'
    # wavs_path = '../../MELD.Raw/dev/dev_wavs'
    # Xy = create_dataset_split(train_csv_path, videos_path, wavs_path)


    train_datasets = []
    test_datasets = []
    #train_datasets.append(MeldDataset('../../MELD.Raw/train/train_sent_emo.csv', split='train'))
    train_datasets.append(MeldDataset('../../MELD.Raw/dev/dev_sent_emo.csv', split='dev', shuffle=True, arbitrary_size=0.5))
    test_datasets.append(MeldDataset('../../MELD.Raw/dev/dev_sent_emo.csv', split='dev', arbitrary_size=0.1))

    emb_ex = test_datasets[0][0]
    print(emb_ex)
    print()
