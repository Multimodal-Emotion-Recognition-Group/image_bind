import logging
from cProfile import label

import numpy as np
import torch
import data

from models import imagebind_model
from models.imagebind_model import ModalityType, load_module
from models import lora as LoRA
from datasets.meld import MeldDataset


if __name__ == '__main__':
    lora = False
    linear_probing = True
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    load_head_post_proc_finetuned = True

    assert not (linear_probing and lora), \
        "Linear probing is a subset of LoRA training procedure for ImageBind. " \
        "Cannot set both linear_probing=True and lora=True. "

    if lora and not load_head_post_proc_finetuned:
        # Hack: adjust lora_factor to the `max batch size used during training / temperature` to compensate missing norm
        lora_factor = 12 / 0.07
    else:
        # This assumes proper loading of all params but results in shift from original dist in case of LoRA
        lora_factor = 1


    # test_dataset = MeldDataset(csv_path='../MELD.Raw/test/test_sent_emo.csv', split='test', for_testing=True, get_audio=False,
    #                            arbitrary_size=0.1, device=device)
    test_dataset = MeldDataset(csv_path='../MELD.Raw/dev/dev_sent_emo.csv', split='dev', for_testing=True,
                               get_audio=False, arbitrary_size=0.1, device=device)

    emotion_labels = test_dataset.classes
    id2label = {k: v for k, v in enumerate(emotion_labels)}
    label2id = {v: k for k, v in enumerate(emotion_labels)}

    # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    if lora:
        model.modality_trunks.update(
            LoRA.apply_lora_modality_trunks(model.modality_trunks, rank=4,
                                            # layer_idxs={ModalityType.TEXT: [0, 1, 2, 3, 4, 5, 6, 7, 8],
                                            #             ModalityType.VISION: [0, 1, 2, 3, 4, 5, 6, 7, 8]},
                                            modality_names=[ModalityType.TEXT, ModalityType.VISION]))

        # Load LoRA params if found
        LoRA.load_lora_modality_trunks(model.modality_trunks,
                                       checkpoint_dir=".checkpoints/lora/550_epochs_lora", postfix="_dreambooth_last")

        if load_head_post_proc_finetuned:
            # Load postprocessors & heads
            load_module(model.modality_postprocessors, module_name="postprocessors",
                        checkpoint_dir=".checkpoints/lora/550_epochs_lora", postfix="_dreambooth_last")
            load_module(model.modality_heads, module_name="heads",
                        checkpoint_dir=".checkpoints/lora/550_epochs_lora", postfix="_dreambooth_last")
    elif linear_probing:
        # Load heads
        load_module(model.modality_heads, module_name="heads",
                    checkpoint_dir="./.checkpoints/lora/22_epochs_MELD_lp", postfix="_MELD_last")

    model.eval()
    model.to(device)

    for i in range(len(test_dataset)):
        # Load data

        images_a, images_class, text_b, text_class, true_label = test_dataset[i]
        labels_tokens = data.load_and_transform_text(emotion_labels, device)

        inputs = {
            ModalityType.TEXT: torch.cat((labels_tokens, text_b), dim=0),
            ModalityType.VISION: images_a
        }

        with torch.no_grad():
            embeddings = model(inputs)

        text_embeddings = embeddings[ModalityType.TEXT]
        vision_embeddings = embeddings[ModalityType.VISION]

        emotion_embeddings = text_embeddings[:len(emotion_labels), :]
        utterance_embedding = text_embeddings[len(emotion_labels), :].unsqueeze(0)  # [1, D]

        text_similarities = (utterance_embedding @ emotion_embeddings.T) * (lora_factor if lora else 1)
        probabilities = torch.softmax(text_similarities, dim=-1).squeeze().cpu().numpy()
        pred_label = id2label[np.argmax(probabilities)]

        vision_similarities = (vision_embeddings @ emotion_embeddings.T) * (lora_factor if lora else 1)
        probabilities = torch.softmax(vision_similarities, dim=-1).squeeze().cpu().numpy()
        pred_label_ = id2label[np.argmax(probabilities)]

        print(f'Utterance X emotion embeddings:\t pred. label: {pred_label}, true label: {true_label}')
        print(f'Video X Emotion embeddings:\t pred. label: {pred_label_}, true label: {true_label}')
        print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')

        if i > 10:
            break
        