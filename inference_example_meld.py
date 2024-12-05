import logging
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


    test_dataset = MeldDataset(csv_path='../MELD.Raw/test/test_sent_emo.csv', split='test', for_testing=True, get_audio=False,
                               arbitrary_size=0.1, device=device)
    emotion_labels = test_dataset.classes

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

        images_a, images_class, text_b, text_class = test_dataset[i]
        # getting rid of 'emotion' label
        # text_b is e.g. 'emotion_label: utterance_text.'
        vision_x_label_inputs = {
            images_class: images_a,
            text_class: text_b,
        }
        utt_x_label_inputs = {

        }

        with torch.no_grad():
            embeddings = model(inputs)

        print(
            "Vision x Text: ",
            torch.softmax(embeddings[images_class] @ embeddings[text_class].T * (lora_factor if lora else 1), dim=-1),
        )

        break
        