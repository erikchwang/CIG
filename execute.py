from main import *
from utils import *
from PIL import ImageDraw, ImageFont


def draw_text(
        config,
        font,
        text,
        is_high
):
    x = 5
    y = 5
    width = config["image_resolution"] - x * 2
    height = config["image_resolution"] * (2 if is_high else 1) - y * 2
    lines = text.split("\n")
    true_lines = []

    for line in lines:
        if font.getsize(line)[0] <= width:
            true_lines.append(line)

        else:
            current_line = ""

            for word in line.split(" "):
                if font.getsize(current_line + word)[0] <= width:
                    current_line += word + " "

                else:
                    true_lines.append(current_line)
                    current_line = word + " "

            true_lines.append(current_line)

    y += height // 2
    line_height = font.getsize(true_lines[0])[1] * 1.5
    y_offset = - (len(true_lines) * line_height) / 2

    text_image = Image.new(
        "RGB",
        (
            config["image_resolution"],
            config["image_resolution"] * (2 if is_high else 1)
        ),
        "white"
    )

    image_draw = ImageDraw.Draw(text_image)

    for line in true_lines:
        image_draw.text(
            (x, y + y_offset),
            line,
            "black",
            font
        )

        y_offset += line_height

    return text_image


def draw_cig_sequence(
        config,
        device,
        font,
        cig_model,
        evaluation_dataset,
        cig_sequence
):
    image_sequences = [
        [
            draw_text(
                config,
                font,
                text,
                text == "Modification Texts:"
            )
        ]
        for text in [
            "Reference Images:",
            "Modification Texts:",
            "Ground-truth Target Images:",
            "Generated Target Images:"
        ]
    ]

    reference_encoder = cig_model.matching_model.reference_encoder
    modification_encoder = cig_model.matching_model.modification_encoder
    reference_feature = None

    for index in cig_sequence:
        reference_image, modification_text, target_image = evaluation_dataset[index]

        if reference_feature is None:
            reference_feature = reference_encoder.extract([reference_image])
            reference_feature = reference_feature.to(device)

        modification_token, modification_mask = modification_encoder.tokenize([modification_text])
        modification_token = modification_token.to(device)
        modification_mask = modification_mask.to(device)

        generated_target_feature = cig_model.generate(
            reference_feature,
            modification_token,
            modification_mask
        )

        reference_feature = torchvision.transforms.functional.resize(
            generated_target_feature,
            reference_feature.shape[2:]
        )

        image_sequences[0].append(
            reference_image.resize(
                (
                    config["image_resolution"],
                    config["image_resolution"]
                )
            )
        )

        image_sequences[1].append(
            draw_text(
                config,
                font,
                modification_text,
                True
            )
        )

        image_sequences[2].append(
            target_image.resize(
                (
                    config["image_resolution"],
                    config["image_resolution"]
                )
            )
        )

        image_sequences[3].append(
            torchvision.transforms.functional.to_pil_image(
                generated_target_feature[0]
            )
        )

    column_count = len(cig_sequence) + 1
    row_count = len(image_sequences)
    margin_width = 5

    cig_sequence_image = Image.new(
        "RGB",
        (
            config["image_resolution"] * column_count + margin_width * (column_count - 1),
            config["image_resolution"] * (row_count + 1) + margin_width * (row_count - 1)
        ),
        "white"
    )

    for column_index in range(column_count):
        for row_index in range(row_count):
            column_offset = (config["image_resolution"] + margin_width) * column_index
            row_offset = (config["image_resolution"] + margin_width) * row_index

            if row_index > 1:
                row_offset += config["image_resolution"]

            cig_sequence_image.paste(
                image_sequences[row_index][column_index],
                (column_offset, row_offset)
            )

    return cig_sequence_image


if __name__ == "__main__":
    if not glob.glob(demo_path):
        os.mkdir(demo_path)

    config = load_yaml(config_path)
    config["stage"] = "inference"

    if torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")

    font = ImageFont.truetype(
        font_path,
        15
    )

    cig_model, _, _, _, _, evaluation_dataset, _ = build(
        config,
        device
    )

    release = torch.load(
        release_path,
        device
    )

    cig_model.load_state_dict(release)
    del release
    cig_model.eval()

    for index, cig_sequence in enumerate(tqdm.tqdm(evaluation_dataset.cig_sequences)):
        with torch.autograd.no_grad():
            cig_sequence_image = draw_cig_sequence(
                config,
                device,
                font,
                cig_model,
                evaluation_dataset,
                cig_sequence
            )

        cig_sequence_image.save(
            os.path.join(
                demo_path,
                "{}.png".format(index)
            )
        )
