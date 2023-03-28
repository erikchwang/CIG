import math
import numpy as np
import torch


def noise_schedule(time):
    result = math.pow(
        math.cos((time + 0.008) / 1.008 * math.pi / 2.0),
        2.0
    )

    return result


def extract_array(
        array,
        index,
        shape
):
    result = torch.index_select(
        torch.tensor(
            array,
            dtype=torch.long if np.issubdtype(array.dtype, np.integer) else torch.float,
            device=index.device
        ),
        0,
        index
    )

    while result.dim() < len(shape):
        result = torch.unsqueeze(
            result,
            result.dim()
        )

    result = torch.broadcast_to(
        result,
        shape
    )

    return result


def normal_kl(
        real_mean,
        real_log_variance,
        model_mean,
        model_log_variance
):
    result = torch.mul(
        torch.sub(
            torch.add(
                torch.sub(
                    torch.add(
                        torch.mul(
                            torch.pow(
                                torch.sub(
                                    real_mean,
                                    model_mean
                                ),
                                2.0
                            ),
                            torch.exp(
                                torch.neg(
                                    model_log_variance
                                )
                            )
                        ),
                        torch.exp(
                            torch.sub(
                                real_log_variance,
                                model_log_variance
                            )
                        )
                    ),
                    real_log_variance
                ),
                model_log_variance
            ),
            1.0
        ),
        0.5
    )

    return result


def standard_normal_cdf(feature):
    result = torch.mul(
        torch.add(
            torch.tanh(
                torch.mul(
                    torch.add(
                        torch.mul(
                            torch.pow(
                                feature,
                                3.0
                            ),
                            0.044715
                        ),
                        feature
                    ),
                    math.sqrt(2.0 / math.pi)
                )
            ),
            1.0
        ),
        0.5
    )

    return result


def normal_nll(
        feature,
        model_mean,
        model_log_variance
):
    centered_feature = torch.sub(
        feature,
        model_mean
    )

    reciprocal_standard_deviation = torch.exp(
        torch.mul(
            model_log_variance,
            -0.5
        )
    )

    cdf_plus = standard_normal_cdf(
        torch.mul(
            torch.add(
                centered_feature,
                1.0 / 255.0
            ),
            reciprocal_standard_deviation
        )
    )

    cdf_minus = standard_normal_cdf(
        torch.mul(
            torch.sub(
                centered_feature,
                1.0 / 255.0
            ),
            reciprocal_standard_deviation
        )
    )

    result = torch.neg(
        torch.where(
            torch.lt(
                feature,
                -0.999
            ),
            torch.log(
                torch.clamp(
                    cdf_plus,
                    1e-12
                )
            ),
            torch.where(
                torch.gt(
                    feature,
                    0.999
                ),
                torch.log(
                    torch.clamp(
                        torch.sub(
                            1.0,
                            cdf_minus
                        ),
                        1e-12
                    )
                ),
                torch.log(
                    torch.clamp(
                        torch.sub(
                            cdf_plus,
                            cdf_minus
                        ),
                        1e-12
                    )
                )
            )
        )
    )

    return result


class DiffusionConfig:
    def __init__(
            self,
            time_step_count,
            time_step_map,
            beta
    ):
        self.time_step_count = time_step_count
        self.time_step_map = time_step_map
        self.log_beta = np.log(beta)
        self.alpha_bar = np.cumprod(1.0 - beta)
        self.sqrt_alpha_bar = np.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = np.sqrt(1.0 - self.alpha_bar)
        self.sqrt_reciprocal_alpha_bar = np.sqrt(1.0 / self.alpha_bar)
        self.sqrt_reciprocal_alpha_bar_minus_one = np.sqrt(1.0 / self.alpha_bar - 1.0)

        last_alpha_bar = np.append(
            1.0,
            self.alpha_bar[:-1]
        )

        self.q_posterior_mean_coefficient_0 = beta * np.sqrt(last_alpha_bar) / (1.0 - self.alpha_bar)
        self.q_posterior_mean_coefficient_1 = np.sqrt(1.0 - beta) * (1.0 - last_alpha_bar) / (1.0 - self.alpha_bar)
        self.q_posterior_variance = beta * (1.0 - last_alpha_bar) / (1.0 - self.alpha_bar)

        self.q_posterior_log_variance = np.log(
            np.append(
                self.q_posterior_variance[1],
                self.q_posterior_variance[1:]
            )
        )

    @classmethod
    def create(
            cls,
            time_step_count
    ):
        diffusion_config = cls(
            time_step_count,
            None,
            np.array(
                [
                    min(
                        1 - noise_schedule((index + 1) / time_step_count) / noise_schedule(index / time_step_count),
                        0.999
                    )
                    for index in range(time_step_count)
                ],
                float
            )
        )

        return diffusion_config

    def convert(
            self,
            sampling_time_step_count
    ):
        time_step_map = np.array(
            [
                round(index * (self.time_step_count - 1) / (sampling_time_step_count - 1))
                for index in range(sampling_time_step_count)
            ],
            int
        )

        alpha_bar = self.alpha_bar[time_step_map]

        last_alpha_bar = np.append(
            1.0,
            alpha_bar[:-1]
        )

        sampling_diffusion_config = DiffusionConfig(
            sampling_time_step_count,
            time_step_map,
            1.0 - alpha_bar / last_alpha_bar
        )

        return sampling_diffusion_config


class DiffusionModel(torch.nn.Module):
    def __init__(
            self,
            denoising_unet,
            time_step_count,
            sampling_time_step_count,
            composition_dropout_rate,
            composition_guidance_scale,
            matching_guidance_scale,
            vlb_loss_scale
    ):
        super().__init__()
        self.denoising_unet = denoising_unet
        self.composition_dropout_rate = composition_dropout_rate
        self.composition_guidance_scale = composition_guidance_scale
        self.matching_guidance_scale = matching_guidance_scale
        self.vlb_loss_scale = vlb_loss_scale
        self.diffusion_config = DiffusionConfig.create(time_step_count)
        self.sampling_diffusion_config = self.diffusion_config.convert(sampling_time_step_count)

    def get_q_sample(
            self,
            target_feature,
            time_step,
            noise
    ):
        q_sample = torch.add(
            torch.mul(
                extract_array(
                    self.diffusion_config.sqrt_alpha_bar,
                    time_step,
                    target_feature.shape
                ),
                target_feature
            ),
            torch.mul(
                extract_array(
                    self.diffusion_config.sqrt_one_minus_alpha_bar,
                    time_step,
                    noise.shape
                ),
                noise
            )
        )

        return q_sample

    def get_q_posterior_distribution(
            self,
            target_feature,
            noisy_target_feature,
            time_step,
            is_sampling
    ):
        if is_sampling:
            diffusion_config = self.sampling_diffusion_config

        else:
            diffusion_config = self.diffusion_config

        q_posterior_mean = torch.add(
            torch.mul(
                extract_array(
                    diffusion_config.q_posterior_mean_coefficient_0,
                    time_step,
                    target_feature.shape
                ),
                target_feature
            ),
            torch.mul(
                extract_array(
                    diffusion_config.q_posterior_mean_coefficient_1,
                    time_step,
                    noisy_target_feature.shape
                ),
                noisy_target_feature
            )
        )

        q_posterior_variance = extract_array(
            diffusion_config.q_posterior_variance,
            time_step,
            target_feature.shape
        )

        q_posterior_log_variance = extract_array(
            diffusion_config.q_posterior_log_variance,
            time_step,
            target_feature.shape
        )

        return q_posterior_mean, q_posterior_variance, q_posterior_log_variance

    def get_denoising_output(
            self,
            noisy_target_feature,
            time_step,
            composition_code,
            composition_spread_code,
            reference_feature,
            is_sampling
    ):
        if is_sampling:
            time_step = extract_array(
                self.sampling_diffusion_config.time_step_map,
                time_step,
                time_step.shape
            )

        else:
            if self.composition_dropout_rate:
                is_dropout = torch.lt(
                    torch.rand(
                        [noisy_target_feature.shape[0]],
                        dtype=torch.float,
                        device=noisy_target_feature.device
                    ),
                    self.composition_dropout_rate
                )

                composition_code = torch.masked_fill(
                    composition_code,
                    torch.unsqueeze(
                        is_dropout,
                        1
                    ),
                    0.0
                )

                composition_spread_code = torch.masked_fill(
                    composition_spread_code,
                    torch.unsqueeze(
                        torch.unsqueeze(
                            is_dropout,
                            1
                        ),
                        2
                    ),
                    0.0
                )

        predicted_noise, predicted_fraction = torch.chunk(
            self.denoising_unet(
                noisy_target_feature,
                time_step,
                composition_code,
                composition_spread_code,
                reference_feature
            ),
            2,
            1
        )

        return predicted_noise, predicted_fraction

    def get_p_distribution(
            self,
            noisy_target_feature,
            time_step,
            composition_code,
            composition_spread_code,
            reference_feature,
            denoising_output,
            is_sampling
    ):
        if is_sampling:
            diffusion_config = self.sampling_diffusion_config
            lower_bound = -1.0
            upper_bound = 1.0

        else:
            diffusion_config = self.diffusion_config
            lower_bound = -float("inf")
            upper_bound = float("inf")

        if denoising_output is None:
            predicted_noise, predicted_fraction = self.get_denoising_output(
                noisy_target_feature,
                time_step,
                composition_code,
                composition_spread_code,
                reference_feature,
                is_sampling
            )

            if self.composition_guidance_scale:
                predicted_noise = torch.add(
                    torch.mul(
                        predicted_noise,
                        self.composition_guidance_scale
                    ),
                    torch.mul(
                        self.get_denoising_output(
                            noisy_target_feature,
                            time_step,
                            torch.zeros_like(composition_code),
                            torch.zeros_like(composition_spread_code),
                            reference_feature,
                            is_sampling
                        )[0],
                        1.0 - self.composition_guidance_scale
                    )
                )

        else:
            predicted_noise, predicted_fraction = denoising_output

        predicted_fraction = torch.mul(
            torch.add(
                predicted_fraction,
                1.0
            ),
            0.5
        )

        p_log_variance = torch.add(
            torch.mul(
                extract_array(
                    diffusion_config.log_beta,
                    time_step,
                    predicted_fraction.shape
                ),
                predicted_fraction
            ),
            torch.mul(
                extract_array(
                    diffusion_config.q_posterior_log_variance,
                    time_step,
                    predicted_fraction.shape
                ),
                torch.sub(
                    1.0,
                    predicted_fraction
                )
            )
        )

        p_variance = torch.exp(p_log_variance)

        p_mean, _, _ = self.get_q_posterior_distribution(
            torch.clamp(
                torch.sub(
                    torch.mul(
                        extract_array(
                            diffusion_config.sqrt_reciprocal_alpha_bar,
                            time_step,
                            noisy_target_feature.shape
                        ),
                        noisy_target_feature
                    ),
                    torch.mul(
                        extract_array(
                            diffusion_config.sqrt_reciprocal_alpha_bar_minus_one,
                            time_step,
                            predicted_noise.shape
                        ),
                        predicted_noise
                    )
                ),
                lower_bound,
                upper_bound
            ),
            noisy_target_feature,
            time_step,
            is_sampling
        )

        return p_mean, p_variance, p_log_variance

    def get_p_sample(
            self,
            noisy_target_feature,
            time_step,
            composition_code,
            composition_spread_code,
            reference_feature,
            matching_model
    ):
        p_mean, p_variance, p_log_variance = self.get_p_distribution(
            noisy_target_feature,
            time_step,
            composition_code,
            composition_spread_code,
            reference_feature,
            None,
            True
        )

        if matching_model is not None and self.matching_guidance_scale:
            p_mean = torch.add(
                torch.mul(
                    torch.mul(
                        matching_model.get_matching_gradient(
                            composition_code,
                            noisy_target_feature
                        ),
                        self.matching_guidance_scale
                    ),
                    p_variance
                ),
                p_mean
            )

        p_sample = torch.add(
            torch.masked_fill(
                torch.mul(
                    torch.exp(
                        torch.mul(
                            p_log_variance,
                            0.5
                        )
                    ),
                    torch.randn_like(noisy_target_feature)
                ),
                torch.unsqueeze(
                    torch.unsqueeze(
                        torch.unsqueeze(
                            torch.eq(
                                time_step,
                                0
                            ),
                            1
                        ),
                        2
                    ),
                    3
                ),
                0.0
            ),
            p_mean
        )

        return p_sample

    def get_p_final_sample(
            self,
            sample_shape,
            composition_code,
            composition_spread_code,
            reference_feature,
            matching_model
    ):
        p_final_sample = torch.randn(
            sample_shape,
            dtype=torch.float,
            device=composition_code.device
        )

        for index in range(self.sampling_diffusion_config.time_step_count - 1, -1, -1):
            p_final_sample = self.get_p_sample(
                p_final_sample,
                torch.full(
                    [p_final_sample.shape[0]],
                    index,
                    dtype=torch.long,
                    device=p_final_sample.device
                ),
                composition_code,
                composition_spread_code,
                reference_feature,
                matching_model
            )

        return p_final_sample

    def forward(
            self,
            target_feature,
            noisy_target_feature,
            time_step,
            composition_code,
            composition_spread_code,
            reference_feature,
            noise
    ):
        predicted_noise, predicted_fraction = self.get_denoising_output(
            noisy_target_feature,
            time_step,
            composition_code,
            composition_spread_code,
            reference_feature,
            False
        )

        diffusion_loss = torch.nn.functional.mse_loss(
            predicted_noise,
            noise
        )

        if self.vlb_loss_scale:
            q_posterior_mean, _, q_posterior_log_variance = self.get_q_posterior_distribution(
                target_feature,
                noisy_target_feature,
                time_step,
                False
            )

            p_mean, _, p_log_variance = self.get_p_distribution(
                noisy_target_feature,
                time_step,
                composition_code,
                composition_spread_code,
                reference_feature,
                (
                    torch.detach(predicted_noise),
                    predicted_fraction
                ),
                False
            )

            diffusion_loss = torch.add(
                torch.mul(
                    torch.mean(
                        torch.where(
                            torch.unsqueeze(
                                torch.unsqueeze(
                                    torch.unsqueeze(
                                        torch.eq(
                                            time_step,
                                            0
                                        ),
                                        1
                                    ),
                                    2
                                ),
                                3
                            ),
                            normal_nll(
                                target_feature,
                                p_mean,
                                p_log_variance
                            ),
                            normal_kl(
                                q_posterior_mean,
                                q_posterior_log_variance,
                                p_mean,
                                p_log_variance
                            )
                        )
                    ),
                    self.vlb_loss_scale / math.log(2.0)
                ),
                diffusion_loss
            )

        return diffusion_loss
