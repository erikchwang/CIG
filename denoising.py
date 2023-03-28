import math
import torch


def zero_module(module):
    for parameter in module.parameters():
        parameter = torch.detach(parameter)
        parameter.zero_()

    return module


def embed_index(
        index,
        size
):
    result = torch.mul(
        torch.unsqueeze(
            torch.exp(
                torch.mul(
                    torch.arange(
                        size // 2,
                        dtype=torch.float,
                        device=index.device
                    ),
                    -2.0 * math.log(10000.0) / size
                )
            ),
            0
        ),
        torch.unsqueeze(
            index.float(),
            1
        )
    )

    result = torch.cat(
        [
            torch.cos(result),
            torch.sin(result)
        ],
        1
    )

    if size % 2:
        result = torch.cat(
            [
                result,
                result[:, -1:]
            ],
            1
        )

    return result


class ResidualBlock(torch.nn.Module):
    def __init__(
            self,
            input_channel_count,
            output_channel_count,
            residual_dropout_rate,
            space_size,
            is_up_sampling,
            is_down_sampling
    ):
        super().__init__()
        self.is_up_sampling = is_up_sampling
        self.is_down_sampling = is_down_sampling

        self.transform_layers = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(
                space_size,
                output_channel_count * 2
            )
        )

        self.front_layers = torch.nn.Sequential(
            torch.nn.GroupNorm(
                32,
                input_channel_count
            ),
            torch.nn.SiLU(),
            torch.nn.Conv2d(
                input_channel_count,
                output_channel_count,
                3,
                padding=1
            )
        )

        self.back_layers = torch.nn.Sequential(
            torch.nn.GroupNorm(
                32,
                output_channel_count
            ),
            torch.nn.SiLU(),
            torch.nn.Dropout(residual_dropout_rate),
            zero_module(
                torch.nn.Conv2d(
                    output_channel_count,
                    output_channel_count,
                    3,
                    padding=1
                )
            )
        )

        if input_channel_count == output_channel_count:
            self.skip_layer = torch.nn.Identity()

        else:
            self.skip_layer = torch.nn.Conv2d(
                input_channel_count,
                output_channel_count,
                1
            )

    def resize(
            self,
            input_feature
    ):
        if self.is_up_sampling and not self.is_down_sampling:
            output_feature = torch.nn.functional.interpolate(
                input_feature,
                scale_factor=2.0
            )

        elif self.is_down_sampling and not self.is_up_sampling:
            output_feature = torch.nn.functional.avg_pool2d(
                input_feature,
                2
            )

        else:
            output_feature = input_feature

        return output_feature

    def _forward(
            self,
            input_feature,
            residual_condition
    ):
        transform_scale, transform_shift = torch.chunk(
            torch.unsqueeze(
                torch.unsqueeze(
                    self.transform_layers(residual_condition),
                    2
                ),
                3
            ),
            2,
            1
        )

        output_feature = torch.add(
            self.back_layers[1:](
                torch.add(
                    torch.mul(
                        self.back_layers[0](
                            self.front_layers[-1](
                                self.resize(
                                    self.front_layers[:-1](
                                        input_feature
                                    )
                                )
                            )
                        ),
                        torch.add(
                            transform_scale,
                            1.0
                        )
                    ),
                    transform_shift
                )
            ),
            self.skip_layer(
                self.resize(
                    input_feature
                )
            )
        )

        return output_feature

    def forward(
            self,
            input_feature,
            residual_condition
    ):
        output_feature = torch.utils.checkpoint.checkpoint(
            self._forward,
            input_feature,
            residual_condition
        )

        return output_feature


class AttentionBlock(torch.nn.Module):
    def __init__(
            self,
            channel_count,
            attention_head_size,
            space_size
    ):
        super().__init__()
        self.attention_head_size = attention_head_size
        self.attention_head_count = channel_count // self.attention_head_size

        self.transform_layer = torch.nn.Conv1d(
            space_size,
            channel_count * 2,
            1
        )

        self.front_layers = torch.nn.Sequential(
            torch.nn.GroupNorm(
                32,
                channel_count
            ),
            torch.nn.Conv1d(
                channel_count,
                channel_count * 3,
                1
            )
        )

        self.back_layer = zero_module(
            torch.nn.Conv1d(
                channel_count,
                channel_count,
                1
            )
        )

    def attend(
            self,
            input_feature,
            attention_condition
    ):
        query_feature, key_feature, value_feature = torch.chunk(
            torch.reshape(
                input_feature,
                [
                    input_feature.shape[0] * self.attention_head_count,
                    self.attention_head_size * 3,
                    input_feature.shape[2]
                ]
            ),
            3,
            1
        )

        key_condition, value_condition = torch.chunk(
            torch.reshape(
                attention_condition,
                [
                    attention_condition.shape[0] * self.attention_head_count,
                    self.attention_head_size * 2,
                    attention_condition.shape[2]
                ]
            ),
            2,
            1
        )

        key_feature = torch.cat(
            [
                key_feature,
                key_condition
            ],
            2
        )

        value_feature = torch.cat(
            [
                value_feature,
                value_condition
            ],
            2
        )

        feature_scale = math.pow(
            self.attention_head_size,
            -0.25
        )

        output_feature = torch.reshape(
            torch.einsum(
                "bts,bcs->bct",
                torch.nn.functional.softmax(
                    torch.einsum(
                        "bct,bcs->bts",
                        torch.mul(
                            query_feature,
                            feature_scale
                        ),
                        torch.mul(
                            key_feature,
                            feature_scale
                        )
                    ),
                    2
                ),
                value_feature
            ),
            [
                input_feature.shape[0],
                self.attention_head_count * self.attention_head_size,
                input_feature.shape[2]
            ]
        )

        return output_feature

    def _forward(
            self,
            input_feature,
            attention_condition
    ):
        output_feature = torch.add(
            torch.reshape(
                self.back_layer(
                    self.attend(
                        self.front_layers[1](
                            torch.reshape(
                                self.front_layers[0](input_feature),
                                [
                                    input_feature.shape[0],
                                    input_feature.shape[1],
                                    input_feature.shape[2] * input_feature.shape[3]
                                ]
                            )
                        ),
                        self.transform_layer(attention_condition)
                    )
                ),
                input_feature.shape
            ),
            input_feature
        )

        return output_feature

    def forward(
            self,
            input_feature,
            attention_condition
    ):
        output_feature = torch.utils.checkpoint.checkpoint(
            self._forward,
            input_feature,
            attention_condition
        )

        return output_feature


class DenoisingSection(torch.nn.Sequential):
    def forward(
            self,
            input_feature,
            residual_condition,
            attention_condition
    ):
        output_feature = input_feature

        for block in self:
            if isinstance(block, ResidualBlock):
                output_feature = block(
                    output_feature,
                    residual_condition
                )

            elif isinstance(block, AttentionBlock):
                output_feature = block(
                    output_feature,
                    attention_condition
                )

            else:
                output_feature = block(output_feature)

        return output_feature


class DenoisingUNet(torch.nn.Module):
    def __init__(
            self,
            input_channel_count,
            output_channel_count,
            unit_channel_count,
            residual_dropout_rate,
            attention_head_size,
            attention_resolutions,
            level_section_count,
            level_multipliers,
            image_resolution,
            space_size
    ):
        super().__init__()
        self.unit_channel_count = unit_channel_count

        self.transform_layers = torch.nn.Sequential(
            torch.nn.Linear(
                self.unit_channel_count,
                space_size
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(
                space_size,
                space_size
            )
        )

        channel_count = self.unit_channel_count * level_multipliers[0]

        self.front_sections = torch.nn.ModuleList(
            [
                DenoisingSection(
                    torch.nn.Conv2d(
                        input_channel_count,
                        channel_count,
                        3,
                        padding=1
                    )
                )
            ]
        )

        front_channel_counts = [channel_count]
        resolution = image_resolution

        for level_index, level_multiplier in enumerate(level_multipliers):
            level_channel_count = self.unit_channel_count * level_multiplier

            for _ in range(level_section_count - 1):
                blocks = [
                    ResidualBlock(
                        channel_count,
                        level_channel_count,
                        residual_dropout_rate,
                        space_size,
                        False,
                        False
                    )
                ]

                channel_count = level_channel_count

                if resolution in attention_resolutions:
                    blocks.append(
                        AttentionBlock(
                            channel_count,
                            attention_head_size,
                            space_size
                        )
                    )

                self.front_sections.append(
                    DenoisingSection(
                        *blocks
                    )
                )

                front_channel_counts.append(channel_count)

            if level_index < len(level_multipliers) - 1:
                self.front_sections.append(
                    DenoisingSection(
                        ResidualBlock(
                            channel_count,
                            channel_count,
                            residual_dropout_rate,
                            space_size,
                            False,
                            True
                        )
                    )
                )

                front_channel_counts.append(channel_count)
                resolution //= 2

        self.middle_section = DenoisingSection(
            ResidualBlock(
                channel_count,
                channel_count,
                residual_dropout_rate,
                space_size,
                False,
                False
            ),
            AttentionBlock(
                channel_count,
                attention_head_size,
                space_size
            ),
            ResidualBlock(
                channel_count,
                channel_count,
                residual_dropout_rate,
                space_size,
                False,
                False
            )
        )

        self.back_sections = torch.nn.ModuleList()

        for level_index, level_multiplier in enumerate(reversed(level_multipliers)):
            level_channel_count = self.unit_channel_count * level_multiplier

            for section_index in range(level_section_count):
                blocks = [
                    ResidualBlock(
                        channel_count + front_channel_counts.pop(),
                        level_channel_count,
                        residual_dropout_rate,
                        space_size,
                        False,
                        False
                    )
                ]

                channel_count = level_channel_count

                if resolution in attention_resolutions:
                    blocks.append(
                        AttentionBlock(
                            channel_count,
                            attention_head_size,
                            space_size
                        )
                    )

                if level_index < len(level_multipliers) - 1 and section_index == level_section_count - 1:
                    blocks.append(
                        ResidualBlock(
                            channel_count,
                            channel_count,
                            residual_dropout_rate,
                            space_size,
                            True,
                            False
                        )
                    )

                    resolution *= 2

                self.back_sections.append(
                    DenoisingSection(
                        *blocks
                    )
                )

        self.prediction_layers = torch.nn.Sequential(
            torch.nn.GroupNorm(
                32,
                channel_count
            ),
            torch.nn.SiLU(),
            zero_module(
                torch.nn.Conv2d(
                    channel_count,
                    output_channel_count,
                    3,
                    padding=1
                )
            )
        )

    def forward(
            self,
            noisy_target_feature,
            time_step,
            composition_code,
            composition_spread_code,
            reference_feature
    ):
        denoising_feature = torch.cat(
            [
                noisy_target_feature,
                reference_feature
            ],
            1
        )

        residual_condition = torch.add(
            self.transform_layers(
                embed_index(
                    time_step,
                    self.unit_channel_count
                )
            ),
            composition_code
        )

        attention_condition = torch.transpose(
            composition_spread_code,
            1,
            2
        )

        front_denoising_features = []

        for section in self.front_sections:
            denoising_feature = section(
                denoising_feature,
                residual_condition,
                attention_condition
            )

            front_denoising_features.append(denoising_feature)

        denoising_feature = self.middle_section(
            denoising_feature,
            residual_condition,
            attention_condition
        )

        for section in self.back_sections:
            denoising_feature = section(
                torch.cat(
                    [
                        denoising_feature,
                        front_denoising_features.pop()
                    ],
                    1
                ),
                residual_condition,
                attention_condition
            )

        predicted_feature = self.prediction_layers(denoising_feature)
        return predicted_feature
