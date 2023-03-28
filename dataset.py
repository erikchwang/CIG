from utils import *
import functools
import torch


class CIGDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            image_base_path,
            split_paths,
            split_multi_turn_paths
    ):
        self.image_base_path = image_base_path

        self.cig_samples = [
            sample
            for path in split_paths
            for sample in load_json(path)
        ]

        if split_multi_turn_paths is None:
            self.cig_sequences = [
                [index]
                for index in range(len(self.cig_samples))
            ]

        else:
            index_map = {
                sample["index"]: index
                for index, sample in enumerate(self.cig_samples)
            }

            self.cig_sequences = sorted(
                [
                    [
                        index_map[index]
                        for index in sequence
                    ]
                    for path in split_multi_turn_paths
                    for sequence in load_json(path)
                ],
                key=len
            )

    def __len__(self):
        length = len(self.cig_samples)
        return length

    def __getitem__(
            self,
            index
    ):
        cig_sample = self.cig_samples[index]

        reference_image = load_image(
            os.path.join(
                self.image_base_path,
                cig_sample["reference_image"]
            )
        )

        modification_text = cig_sample["modification_text"]

        target_image = load_image(
            os.path.join(
                self.image_base_path,
                cig_sample["target_image"]
            )
        )

        return reference_image, modification_text, target_image

    def get_cig_loader(
            self,
            batch_size,
            reference_encoder,
            modification_encoder,
            target_encoder
    ):
        cig_loader = torch.utils.data.DataLoader(
            self,
            batch_size,
            True,
            num_workers=2,
            collate_fn=functools.partial(
                CIGBatch.collate,
                reference_encoder=reference_encoder,
                modification_encoder=modification_encoder,
                target_encoder=target_encoder
            ),
            pin_memory=True,
            drop_last=True
        )

        return cig_loader

    def get_cig_iterators(
            self,
            batch_size,
            reference_encoder,
            modification_encoder,
            target_encoder
    ):
        def cig_iterator(cig_sequences):
            sequence_lengths = [
                len(sequence)
                for sequence in cig_sequences
            ]

            for index in range(max(sequence_lengths)):
                cig_samples = []

                for sequence, length in zip(cig_sequences, sequence_lengths):
                    if index < length:
                        cig_samples.append(self[sequence[index]])

                    else:
                        cig_samples.append(self[sequence[-1]])

                cig_batch = CIGBatch.collate(
                    cig_samples,
                    reference_encoder,
                    modification_encoder,
                    target_encoder
                )

                yield cig_batch

            yield sequence_lengths

        cig_iterators = [
            cig_iterator(self.cig_sequences[offset:offset + batch_size])
            for offset in range(0, len(self.cig_sequences), batch_size)
        ]

        return cig_iterators


class CIGBatch:
    def __init__(
            self,
            reference_feature,
            modification_token,
            modification_mask,
            target_feature
    ):
        self.reference_feature = reference_feature
        self.modification_token = modification_token
        self.modification_mask = modification_mask
        self.target_feature = target_feature

    @classmethod
    def collate(
            cls,
            cig_samples,
            reference_encoder,
            modification_encoder,
            target_encoder
    ):
        reference_images, modification_texts, target_images = zip(*cig_samples)
        reference_feature = reference_encoder.extract(reference_images)
        modification_token, modification_mask = modification_encoder.tokenize(modification_texts)
        target_feature = target_encoder.extract(target_images)

        cig_batch = cls(
            reference_feature,
            modification_token,
            modification_mask,
            target_feature
        )

        return cig_batch

    def pin_memory(self):
        self.reference_feature = self.reference_feature.pin_memory()
        self.modification_token = self.modification_token.pin_memory()
        self.modification_mask = self.modification_mask.pin_memory()
        self.target_feature = self.target_feature.pin_memory()
        return self

    def to(self, *args, **kwargs):
        self.reference_feature = self.reference_feature.to(*args, **kwargs)
        self.modification_token = self.modification_token.to(*args, **kwargs)
        self.modification_mask = self.modification_mask.to(*args, **kwargs)
        self.target_feature = self.target_feature.to(*args, **kwargs)
        return self
