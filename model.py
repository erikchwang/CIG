from denoising import *
from diffusion import *
from matching import *
import transformers


class CIGModel(torch.nn.Module):
    def __init__(
            self,
            clip_version,
            image_resolution,
            space_size,
            initial_temperature,
            unit_channel_count,
            residual_dropout_rate,
            attention_head_size,
            attention_resolutions,
            level_section_count,
            level_multipliers,
            time_step_count,
            sampling_time_step_count,
            composition_dropout_rate,
            composition_guidance_scale,
            matching_guidance_scale,
            vlb_loss_scale
    ):
        super().__init__()
        self.image_resolution = image_resolution

        self.matching_model = MatchingModel(
            ImageEncoder(
                transformers.CLIPFeatureExtractor.from_pretrained(
                    clip_version,
                    image_mean=[0.0, 0.0, 0.0],
                    image_std=[1.0, 1.0, 1.0]
                ),
                transformers.CLIPVisionModel.from_pretrained(clip_version),
                space_size
            ),
            TextEncoder(
                transformers.CLIPTokenizer.from_pretrained(clip_version),
                transformers.CLIPTextModel.from_pretrained(clip_version),
                space_size
            ),
            ImageEncoder(
                transformers.CLIPFeatureExtractor.from_pretrained(
                    clip_version,
                    size=self.image_resolution,
                    crop_size=self.image_resolution,
                    image_mean=[0.0, 0.0, 0.0],
                    image_std=[1.0, 1.0, 1.0]
                ),
                transformers.CLIPVisionModel(
                    transformers.CLIPVisionConfig.from_pretrained(
                        clip_version,
                        image_size=self.image_resolution
                    )
                ),
                space_size
            ),
            FusionModule(space_size),
            initial_temperature
        )

        self.matching_model.reference_encoder.clip_encoder.gradient_checkpointing_enable()
        self.matching_model.modification_encoder.clip_encoder.gradient_checkpointing_enable()
        self.matching_model.target_encoder.clip_encoder.gradient_checkpointing_enable()

        self.diffusion_model = DiffusionModel(
            DenoisingUNet(
                6,
                6,
                unit_channel_count,
                residual_dropout_rate,
                attention_head_size,
                attention_resolutions,
                level_section_count,
                level_multipliers,
                self.image_resolution,
                space_size
            ),
            time_step_count,
            sampling_time_step_count,
            composition_dropout_rate,
            composition_guidance_scale,
            matching_guidance_scale,
            vlb_loss_scale
        )

    def get_cig_toolkit(
            self,
            backbone_activity,
            learning_rate
    ):
        parameters = {
            "backbone": {
                "decay": [],
                "other": []
            },
            "other": {
                "decay": [],
                "other": []
            }
        }

        for module in [
            self.matching_model.reference_encoder.clip_encoder,
            self.matching_model.modification_encoder.clip_encoder
        ]:
            if backbone_activity:
                for name, parameter in module.named_parameters():
                    if any(term in name.lower() for term in ["bias", "norm"]):
                        parameters["backbone"]["other"].append(parameter)

                    else:
                        parameters["backbone"]["decay"].append(parameter)

            else:
                module.requires_grad_(False)

        for module in [
            self.matching_model.reference_encoder.projector,
            self.matching_model.reference_encoder.spread_projector,
            self.matching_model.modification_encoder.projector,
            self.matching_model.modification_encoder.spread_projector,
            self.matching_model.target_encoder,
            self.matching_model.fusion_module
        ]:
            for name, parameter in module.named_parameters():
                if any(term in name.lower() for term in ["bias", "norm"]):
                    parameters["other"]["other"].append(parameter)

                else:
                    parameters["other"]["decay"].append(parameter)

        parameters["other"]["other"].append(self.matching_model.temperature)
        parameters["other"]["other"].extend(self.diffusion_model.parameters())

        cig_optimizer = torch.optim.AdamW(
            [
                {
                    "params": parameters["backbone"]["decay"],
                    "lr": learning_rate * backbone_activity
                },
                {
                    "params": parameters["backbone"]["other"],
                    "lr": learning_rate * backbone_activity,
                    "weight_decay": 0.0
                },
                {
                    "params": parameters["other"]["decay"]
                },
                {
                    "params": parameters["other"]["other"],
                    "weight_decay": 0.0
                }
            ],
            learning_rate
        )

        cig_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            cig_optimizer,
            "min",
            0.5,
            0
        )

        cig_scaler = torch.cuda.amp.GradScaler()
        return cig_optimizer, cig_scheduler, cig_scaler

    def generate(
            self,
            reference_feature,
            modification_token,
            modification_mask
    ):
        composition_code, composition_spread_code = self.matching_model.get_composition_code(
            reference_feature,
            modification_token,
            modification_mask
        )

        generated_target_feature = self.diffusion_model.get_p_final_sample(
            [
                reference_feature.shape[0],
                3,
                self.image_resolution,
                self.image_resolution
            ],
            composition_code,
            composition_spread_code,
            torchvision.transforms.functional.resize(
                reference_feature,
                [
                    self.image_resolution,
                    self.image_resolution
                ]
            ),
            self.matching_model
        )

        return generated_target_feature

    def forward(
            self,
            reference_feature,
            modification_token,
            modification_mask,
            target_feature,
            matching_loss_scale,
            diffusion_loss_scale,
            noisy_matching_loss_scale
    ):
        time_step = torch.randint(
            self.diffusion_model.diffusion_config.time_step_count,
            [reference_feature.shape[0]],
            dtype=torch.long,
            device=reference_feature.device
        )

        noise = torch.randn_like(target_feature)

        noisy_target_feature = self.diffusion_model.get_q_sample(
            target_feature,
            time_step,
            noise
        )

        composition_code, composition_spread_code, matching_loss = self.matching_model(
            reference_feature,
            modification_token,
            modification_mask,
            target_feature if matching_loss_scale else None,
            noisy_target_feature if noisy_matching_loss_scale else None
        )

        cig_loss = torch.mul(
            matching_loss,
            matching_loss_scale if matching_loss_scale else noisy_matching_loss_scale
        )

        if diffusion_loss_scale:
            cig_loss = torch.add(
                torch.mul(
                    self.diffusion_model(
                        target_feature,
                        noisy_target_feature,
                        time_step,
                        composition_code,
                        composition_spread_code,
                        torchvision.transforms.functional.resize(
                            reference_feature,
                            [
                                self.image_resolution,
                                self.image_resolution
                            ]
                        ),
                        noise
                    ),
                    diffusion_loss_scale
                ),
                cig_loss
            )

        return cig_loss
