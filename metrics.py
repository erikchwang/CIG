import itertools
import numpy as np
import torch
import torchvision


class ObjectDetector(torch.nn.Module):
    def __init__(
            self,
            class_count,
            coordinate_size
    ):
        super().__init__()
        self.class_count = class_count
        self.coordinate_size = coordinate_size
        self.inception_model = torchvision.models.inception.Inception3(512)

        self.object_predictor = torch.nn.Sequential(
            torch.nn.Linear(
                512,
                256
            ),
            torch.nn.Linear(
                256,
                self.class_count
            ),
            torch.nn.Sigmoid()
        )

        self.object_localizer = torch.nn.Sequential(
            torch.nn.Linear(
                1024,
                512
            ),
            torch.nn.Linear(
                512,
                self.class_count * self.coordinate_size
            )
        )

    def forward(
            self,
            input_feature
    ):
        inception_feature = torch.add(
            torch.mul(
                input_feature,
                torch.unsqueeze(
                    torch.unsqueeze(
                        torch.unsqueeze(
                            torch.tensor(
                                [0.229 / 0.5, 0.224 / 0.5, 0.225 / 0.5],
                                dtype=torch.float,
                                device=input_feature.device
                            ),
                            0
                        ),
                        2
                    ),
                    3
                )
            ),
            torch.unsqueeze(
                torch.unsqueeze(
                    torch.unsqueeze(
                        torch.tensor(
                            [(0.485 - 0.5) / 0.5, (0.456 - 0.5) / 0.5, (0.406 - 0.5) / 0.5],
                            dtype=torch.float,
                            device=input_feature.device
                        ),
                        0
                    ),
                    2
                ),
                3
            )
        )

        inception_feature = self.inception_model.Conv2d_1a_3x3(inception_feature)
        inception_feature = self.inception_model.Conv2d_2a_3x3(inception_feature)
        inception_feature = self.inception_model.Conv2d_2b_3x3(inception_feature)
        inception_feature = self.inception_model.maxpool1(inception_feature)
        inception_feature = self.inception_model.Conv2d_3b_1x1(inception_feature)
        inception_feature = self.inception_model.Conv2d_4a_3x3(inception_feature)
        inception_feature = self.inception_model.maxpool2(inception_feature)
        inception_feature = self.inception_model.Mixed_5b(inception_feature)
        inception_feature = self.inception_model.Mixed_5c(inception_feature)
        inception_feature = self.inception_model.Mixed_5d(inception_feature)
        inception_feature = self.inception_model.Mixed_6a(inception_feature)
        inception_feature = self.inception_model.Mixed_6b(inception_feature)
        inception_feature = self.inception_model.Mixed_6c(inception_feature)
        inception_feature = self.inception_model.Mixed_6d(inception_feature)
        inception_feature = self.inception_model.Mixed_6e(inception_feature)
        auxiliary_feature = self.inception_model.AuxLogits(inception_feature)
        inception_feature = self.inception_model.Mixed_7a(inception_feature)
        inception_feature = self.inception_model.Mixed_7b(inception_feature)
        inception_feature = self.inception_model.Mixed_7c(inception_feature)
        inception_feature = self.inception_model.avgpool(inception_feature)
        inception_feature = self.inception_model.dropout(inception_feature)

        inception_feature = self.inception_model.fc(
            torch.flatten(
                inception_feature,
                1
            )
        )

        object_prediction = torch.gt(
            self.object_predictor(inception_feature),
            0.5
        )

        object_location = torch.reshape(
            self.object_localizer(
                torch.cat(
                    [
                        inception_feature,
                        auxiliary_feature
                    ],
                    1
                )
            ),
            [
                -1,
                self.class_count,
                self.coordinate_size
            ]
        )

        return object_prediction, object_location


def get_object_detector(
        object_detector_checkpoint_path,
        device
):
    checkpoint = torch.load(
        object_detector_checkpoint_path,
        device
    )

    object_detector = ObjectDetector(
        checkpoint["num_classes"],
        2 if checkpoint["num_classes"] == 24 else 3
    )

    object_detector.to(device)

    object_detector.load_state_dict(
        {
            key: value
            for (key, _), (_, value) in zip(object_detector.state_dict().items(), checkpoint["state_dict"].items())
        }
    )

    del checkpoint
    object_detector.eval()
    return object_detector


def construct_scene_graph(coordinates):
    scene_graph = np.empty(
        [
            2,
            len(coordinates),
            len(coordinates)
        ],
        int
    )

    for reference_index, reference_coordinate in enumerate(coordinates):
        reference_x = reference_coordinate[0]
        reference_y = reference_coordinate[-1]

        for query_index, query_coordinate in enumerate(coordinates):
            if reference_index == query_index:
                query_x = 0.5
                query_y = 0.5

            else:
                query_x = query_coordinate[0]
                query_y = query_coordinate[-1]

            if reference_x > query_x:
                scene_graph[0, reference_index, query_index] = 1

            elif reference_x < query_x:
                scene_graph[0, reference_index, query_index] = -1

            else:
                scene_graph[0, reference_index, query_index] = 0

            if reference_y > query_y:
                scene_graph[1, reference_index, query_index] = 1

            elif reference_y < query_y:
                scene_graph[1, reference_index, query_index] = -1

            else:
                scene_graph[1, reference_index, query_index] = 0

    return scene_graph


def get_graph_similarity(
        generated_target_object_prediction,
        generated_target_object_location,
        target_object_prediction,
        target_object_location
):
    true_positive_prediction = torch.logical_and(
        generated_target_object_prediction,
        target_object_prediction
    )

    true_positive_count = torch.count_nonzero(true_positive_prediction).tolist()

    if true_positive_count:
        generated_target_object_coordinates = generated_target_object_location[true_positive_prediction].cpu().numpy()
        target_object_coordinates = target_object_location[true_positive_prediction].cpu().numpy()
        generated_target_scene_graph = construct_scene_graph(generated_target_object_coordinates)
        target_scene_graph = construct_scene_graph(target_object_coordinates)
        graph_accuracy = np.average(generated_target_scene_graph == target_scene_graph)
        graph_similarity = graph_accuracy * true_positive_count / torch.count_nonzero(target_object_prediction).tolist()

    else:
        graph_similarity = 0.0

    return graph_similarity


def get_object_detection(
        generated_target_sequence_feature,
        target_sequence_feature,
        sequence_lengths,
        object_detector
):
    generated_target_feature = torchvision.transforms.functional.resize(
        torch.cat(
            [
                generated_target_sequence_feature[index, :length]
                for index, length in enumerate(sequence_lengths)
            ],
            0
        ),
        [299, 299]
    )

    target_feature = torchvision.transforms.functional.resize(
        torch.cat(
            [
                target_sequence_feature[index, :length]
                for index, length in enumerate(sequence_lengths)
            ],
            0
        ),
        [299, 299]
    )

    generated_target_object_prediction, generated_target_object_location = object_detector(generated_target_feature)
    target_object_prediction, target_object_location = object_detector(target_feature)

    true_positive_count = torch.count_nonzero(
        torch.logical_and(
            generated_target_object_prediction,
            target_object_prediction
        )
    ).tolist()

    false_positive_count = torch.count_nonzero(
        torch.logical_and(
            generated_target_object_prediction,
            torch.logical_not(target_object_prediction)
        )
    ).tolist()

    false_negative_count = torch.count_nonzero(
        torch.logical_and(
            torch.logical_not(generated_target_object_prediction),
            target_object_prediction
        )
    ).tolist()

    graph_similarities = [
        get_graph_similarity(
            generated_target_object_prediction[length - 1],
            generated_target_object_location[length - 1],
            target_object_prediction[length - 1],
            target_object_location[length - 1]
        )
        for length in itertools.accumulate(sequence_lengths)
    ]

    return true_positive_count, false_positive_count, false_negative_count, graph_similarities
