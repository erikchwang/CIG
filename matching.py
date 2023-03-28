import torch
import torchvision


class ImageEncoder(torch.nn.Module):
    def __init__(
            self,
            clip_extractor,
            clip_encoder,
            space_size
    ):
        super().__init__()
        self.clip_extractor = clip_extractor
        self.clip_encoder = clip_encoder

        self.projector = torch.nn.Linear(
            self.clip_encoder.config.hidden_size,
            space_size
        )

        self.spread_projector = torch.nn.Linear(
            self.clip_encoder.config.hidden_size,
            space_size
        )

    def extract(
            self,
            images
    ):
        extraction = self.clip_extractor(
            list(images),
            return_tensors="pt"
        )

        feature = extraction["pixel_values"]
        return feature

    def forward(
            self,
            feature
    ):
        encoding = self.clip_encoder(feature)
        code = self.projector(encoding.pooler_output)
        spread_code = self.spread_projector(encoding.last_hidden_state)
        return code, spread_code


class TextEncoder(torch.nn.Module):
    def __init__(
            self,
            clip_tokenizer,
            clip_encoder,
            space_size
    ):
        super().__init__()
        self.clip_tokenizer = clip_tokenizer
        self.clip_encoder = clip_encoder

        self.projector = torch.nn.Linear(
            self.clip_encoder.config.hidden_size,
            space_size
        )

        self.spread_projector = torch.nn.Linear(
            self.clip_encoder.config.hidden_size,
            space_size
        )

    def tokenize(
            self,
            texts
    ):
        tokenization = self.clip_tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        token = tokenization["input_ids"]
        mask = tokenization["attention_mask"]
        return token, mask

    def forward(
            self,
            token,
            mask
    ):
        encoding = self.clip_encoder(
            token,
            mask
        )

        code = self.projector(encoding.pooler_output)
        spread_code = self.spread_projector(encoding.last_hidden_state)
        return code, spread_code


class FusionModule(torch.nn.Module):
    def __init__(
            self,
            space_size
    ):
        super().__init__()

        self.gate_layers = torch.nn.Sequential(
            torch.nn.Linear(
                space_size * 4,
                space_size
            ),
            torch.nn.Sigmoid()
        )

        self.transform_layers = torch.nn.Sequential(
            torch.nn.Linear(
                space_size * 4,
                space_size
            ),
            torch.nn.GELU()
        )

    def forward(
            self,
            subject_code,
            object_code
    ):
        joint_code = torch.cat(
            [
                subject_code,
                object_code,
                torch.mul(
                    subject_code,
                    object_code
                ),
                torch.sub(
                    subject_code,
                    object_code
                )
            ],
            1
        )

        gate_scale = self.gate_layers(joint_code)

        fusion_code = torch.add(
            torch.mul(
                self.transform_layers(joint_code),
                gate_scale
            ),
            torch.mul(
                subject_code,
                torch.sub(
                    1.0,
                    gate_scale
                )
            )
        )

        return fusion_code


class MatchingModel(torch.nn.Module):
    def __init__(
            self,
            reference_encoder,
            modification_encoder,
            target_encoder,
            fusion_module,
            initial_temperature
    ):
        super().__init__()
        self.reference_encoder = reference_encoder
        self.modification_encoder = modification_encoder
        self.target_encoder = target_encoder
        self.fusion_module = fusion_module

        self.temperature = torch.nn.parameter.Parameter(
            torch.tensor(
                initial_temperature,
                dtype=torch.float
            )
        )

    def get_composition_code(
            self,
            reference_feature,
            modification_token,
            modification_mask
    ):
        reference_code, reference_spread_code = self.reference_encoder(reference_feature)

        modification_code, modification_spread_code = self.modification_encoder(
            modification_token,
            modification_mask
        )

        composition_code = self.fusion_module(
            reference_code,
            modification_code
        )

        composition_spread_code = torch.cat(
            [
                reference_spread_code,
                modification_spread_code
            ],
            1
        )

        return composition_code, composition_spread_code

    def get_matching_score(
            self,
            composition_code,
            target_code
    ):
        matching_score = torch.mul(
            torch.matmul(
                torch.nn.functional.normalize(composition_code),
                torch.t(
                    torch.nn.functional.normalize(
                        target_code
                    )
                )
            ),
            torch.exp(self.temperature)
        )

        return matching_score

    def get_matching_gradient(
            self,
            composition_code,
            noisy_target_feature
    ):
        with torch.autograd.enable_grad():
            noisy_target_feature = torch.detach(noisy_target_feature)
            noisy_target_feature.requires_grad_(True)

            torch.autograd.backward(
                torch.mean(
                    torch.diagonal(
                        self.get_matching_score(
                            torch.detach(composition_code),
                            self.target_encoder(noisy_target_feature)[0]
                        )
                    )
                )
            )

            matching_gradient = torch.detach(noisy_target_feature.grad)

        return matching_gradient

    def forward(
            self,
            reference_feature,
            modification_token,
            modification_mask,
            target_feature,
            noisy_target_feature
    ):
        composition_code, composition_spread_code = self.get_composition_code(
            reference_feature,
            modification_token,
            modification_mask
        )

        if target_feature is not None and noisy_target_feature is None:
            matching_score = self.get_matching_score(
                composition_code,
                self.reference_encoder(
                    torchvision.transforms.functional.resize(
                        target_feature,
                        reference_feature.shape[2:]
                    )
                )[0]
            )

        elif noisy_target_feature is not None and target_feature is None:
            matching_score = self.get_matching_score(
                torch.detach(composition_code),
                self.target_encoder(noisy_target_feature)[0]
            )

        else:
            matching_score = None

        if matching_score is None:
            matching_loss = 0.0

        else:
            matching_label = torch.arange(
                matching_score.shape[0],
                dtype=torch.long,
                device=matching_score.device
            )

            matching_loss = torch.add(
                torch.nn.functional.cross_entropy(
                    matching_score,
                    matching_label
                ),
                torch.nn.functional.cross_entropy(
                    torch.t(matching_score),
                    matching_label
                )
            )

        return composition_code, composition_spread_code, matching_loss
