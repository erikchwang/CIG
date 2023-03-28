from dataset import *
from metrics import *
from model import *
from utils import *
import argparse
import glob
import tqdm


def build(
        config,
        device
):
    cig_model = CIGModel(
        config["clip_version"],
        config["image_resolution"],
        config["space_size"],
        config["initial_temperature"],
        config["unit_channel_count"],
        config["residual_dropout_rate"],
        config["attention_head_size"],
        config["attention_resolutions"],
        config["level_section_count"],
        config["level_multipliers"],
        config["time_step_count"],
        config["sampling_time_step_count"],
        config["composition_dropout_rate"],
        config["composition_guidance_scale"],
        config["matching_guidance_scale"],
        config["vlb_loss_scale"]
    )

    cig_model.to(device)

    if config["stage"] == "inference":
        cig_optimizer = None
        cig_scheduler = None
        cig_scaler = None
        optimization_dataset = None

        evaluation_dataset = CIGDataset(
            os.path.join(
                root_path,
                config["image_base_path"]
            ),
            [
                os.path.join(
                    root_path,
                    path
                )
                for path in config["test_paths"]
            ],
            [
                os.path.join(
                    root_path,
                    path
                )
                for path in config["test_multi_turn_paths"]
            ]
        )

        object_detector = get_object_detector(
            os.path.join(
                root_path,
                config["object_detector_checkpoint_path"]
            ),
            device
        )

    else:
        cig_optimizer, cig_scheduler, cig_scaler = cig_model.get_cig_toolkit(
            0.0 if config["stage"] == "noisy_matching" else config["backbone_activity"],
            config["learning_rate"]
        )

        optimization_dataset = CIGDataset(
            os.path.join(
                root_path,
                config["image_base_path"]
            ),
            [
                os.path.join(
                    root_path,
                    path
                )
                for path in config["train_paths"]
            ],
            None
        )

        evaluation_dataset = CIGDataset(
            os.path.join(
                root_path,
                config["image_base_path"]
            ),
            [
                os.path.join(
                    root_path,
                    path
                )
                for path in config["develop_paths"]
            ],
            None
        )

        object_detector = None

    return cig_model, cig_optimizer, cig_scheduler, cig_scaler, optimization_dataset, evaluation_dataset, object_detector


def optimize(
        config,
        device,
        cig_model,
        cig_optimizer,
        cig_scaler,
        optimization_dataset
):
    cig_optimizer.zero_grad(True)

    cig_loader = optimization_dataset.get_cig_loader(
        config["batch_size"],
        cig_model.matching_model.reference_encoder,
        cig_model.matching_model.modification_encoder,
        cig_model.matching_model.target_encoder
    )

    for _ in range(1 if config["stage"] == "matching" else config["round_count"]):
        for index, cig_batch in enumerate(tqdm.tqdm(cig_loader)):
            cig_batch.to(device)

            with torch.cuda.amp.autocast():
                cig_loss = torch.div(
                    cig_model(
                        cig_batch.reference_feature,
                        cig_batch.modification_token,
                        cig_batch.modification_mask,
                        cig_batch.target_feature,
                        config["matching_loss_scale"],
                        config["diffusion_loss_scale"],
                        config["noisy_matching_loss_scale"]
                    ),
                    config["gradient_accumulation_span"]
                )

            cig_loss = cig_scaler.scale(cig_loss)
            cig_loss.backward()

            if (index + 1) % config["gradient_accumulation_span"] == 0:
                cig_scaler.step(cig_optimizer)
                cig_scaler.update()
                cig_optimizer.zero_grad(True)


def evaluate(
        config,
        device,
        cig_model,
        evaluation_dataset,
        object_detector
):
    if config["stage"] == "inference":
        cig_iterators = evaluation_dataset.get_cig_iterators(
            config["batch_size"],
            cig_model.matching_model.reference_encoder,
            cig_model.matching_model.modification_encoder,
            cig_model.matching_model.target_encoder
        )

        cumulate_true_positive_count = 0
        cumulate_false_positive_count = 0
        cumulate_false_negative_count = 0
        cumulate_graph_similarities = []

        for cig_iterator in tqdm.tqdm(cig_iterators):
            reference_feature = None
            sequence_lengths = None
            generated_target_features = []
            target_features = []

            for item in cig_iterator:
                if isinstance(item, CIGBatch):
                    cig_batch = item
                    cig_batch.to(device)

                    if reference_feature is None or not config["is_recurrent"]:
                        reference_feature = cig_batch.reference_feature

                    generated_target_feature = cig_model.generate(
                        reference_feature,
                        cig_batch.modification_token,
                        cig_batch.modification_mask
                    )

                    if config["is_recurrent"]:
                        reference_feature = torchvision.transforms.functional.resize(
                            generated_target_feature,
                            reference_feature.shape[2:]
                        )

                    generated_target_features.append(generated_target_feature)
                    target_features.append(cig_batch.target_feature)

                else:
                    sequence_lengths = item

            generated_target_sequence_feature = torch.cat(
                [
                    torch.unsqueeze(
                        feature,
                        1
                    )
                    for feature in generated_target_features
                ],
                1
            )

            target_sequence_feature = torch.cat(
                [
                    torch.unsqueeze(
                        feature,
                        1
                    )
                    for feature in target_features
                ],
                1
            )

            true_positive_count, false_positive_count, false_negative_count, graph_similarities = get_object_detection(
                generated_target_sequence_feature,
                target_sequence_feature,
                sequence_lengths,
                object_detector
            )

            cumulate_true_positive_count += true_positive_count
            cumulate_false_positive_count += false_positive_count
            cumulate_false_negative_count += false_negative_count
            cumulate_graph_similarities.extend(graph_similarities)

        precision = cumulate_true_positive_count / (cumulate_true_positive_count + cumulate_false_positive_count)
        recall = cumulate_true_positive_count / (cumulate_true_positive_count + cumulate_false_negative_count)
        f1 = precision * recall * 2.0 / (precision + recall)
        rsim = sum(cumulate_graph_similarities) / len(cumulate_graph_similarities)

        performance = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "rsim": rsim
        }

    else:
        cig_loader = evaluation_dataset.get_cig_loader(
            config["batch_size"],
            cig_model.matching_model.reference_encoder,
            cig_model.matching_model.modification_encoder,
            cig_model.matching_model.target_encoder
        )

        cig_losses = []

        for _ in range(1 if config["stage"] == "matching" else config["round_count"]):
            for cig_batch in cig_loader:
                cig_batch.to(device)

                cig_losses.append(
                    cig_model(
                        cig_batch.reference_feature,
                        cig_batch.modification_token,
                        cig_batch.modification_mask,
                        cig_batch.target_feature,
                        config["matching_loss_scale"],
                        config["diffusion_loss_scale"],
                        config["noisy_matching_loss_scale"]
                    ).tolist()
                )

        average_cig_loss = sum(cig_losses) / len(cig_losses)
        performance = {"average_cig_loss": average_cig_loss}

    return performance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        nargs="?",
        type=str,
        required=True
    )

    parser.add_argument(
        "--matching_loss_scale",
        nargs="?",
        type=float,
        required=False
    )

    parser.add_argument(
        "--diffusion_loss_scale",
        nargs="?",
        type=float,
        required=False
    )

    parser.add_argument(
        "--noisy_matching_loss_scale",
        nargs="?",
        type=float,
        required=False
    )

    config = load_yaml(config_path)

    config.update(
        {
            key: value
            for key, value in vars(parser.parse_args()).items()
            if value is not None
        }
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")

    cig_model, cig_optimizer, cig_scheduler, cig_scaler, optimization_dataset, evaluation_dataset, object_detector = build(
        config,
        device
    )

    if config["stage"] == "inference":
        release = torch.load(
            release_path,
            device
        )

        archive = load_json(archive_path)
        cig_model.load_state_dict(release)
        del release
        cig_model.eval()

        with torch.autograd.no_grad():
            performance = evaluate(
                config,
                device,
                cig_model,
                evaluation_dataset,
                object_detector
            )

        print(
            "{}: {}.".format(
                config["stage"],
                json.dumps(performance)
            ),
            end=" "
        )

        archive[config["stage"]] = performance

        dump_json(
            archive,
            archive_path
        )

        print("archive saved.")

    else:
        if not glob.glob(outcome_path):
            os.mkdir(outcome_path)

        if glob.glob(checkpoint_path) and glob.glob(archive_path):
            checkpoint = torch.load(
                checkpoint_path,
                device
            )

            archive = load_json(archive_path)
            cig_model.load_state_dict(checkpoint["model_state"])

            if config["stage"] == checkpoint["stage"]:
                cig_optimizer.load_state_dict(checkpoint["optimizer_state"])
                cig_scheduler.load_state_dict(checkpoint["scheduler_state"])
                cig_scaler.load_state_dict(checkpoint["scaler_state"])
                session = checkpoint["session"]

            else:
                session = 0
                archive[config["stage"]] = []

            del checkpoint

        else:
            session = 0
            archive = {config["stage"]: []}

        while session < config["session_count"]:
            cig_model.train()

            optimize(
                config,
                device,
                cig_model,
                cig_optimizer,
                cig_scaler,
                optimization_dataset
            )

            cig_model.eval()

            with torch.autograd.no_grad():
                performance = evaluate(
                    config,
                    device,
                    cig_model,
                    evaluation_dataset,
                    object_detector
                )

            print(
                "epoch {}: {}.".format(
                    cig_scheduler.last_epoch,
                    json.dumps(performance)
                ),
                end=" "
            )

            archive[config["stage"]].append(performance)

            dump_json(
                archive,
                archive_path
            )

            print(
                "archive saved.",
                end=" "
            )

            if cig_scheduler.is_better(performance["average_cig_loss"], cig_scheduler.best):
                torch.save(
                    cig_model.state_dict(),
                    release_path
                )

                print(
                    "release saved.",
                    end=" "
                )

            else:
                checkpoint = torch.load(
                    checkpoint_path,
                    device
                )

                cig_model.load_state_dict(checkpoint["model_state"])
                cig_optimizer.load_state_dict(checkpoint["optimizer_state"])
                cig_scheduler.load_state_dict(checkpoint["scheduler_state"])
                cig_scaler.load_state_dict(checkpoint["scaler_state"])
                del checkpoint
                session += 1

                print(
                    "state restored.",
                    end=" "
                )

            cig_scheduler.step(performance["average_cig_loss"])

            torch.save(
                {
                    "model_state": cig_model.state_dict(),
                    "optimizer_state": cig_optimizer.state_dict(),
                    "scheduler_state": cig_scheduler.state_dict(),
                    "scaler_state": cig_scaler.state_dict(),
                    "stage": config["stage"],
                    "session": session
                },
                checkpoint_path
            )

            print("checkpoint saved.")
