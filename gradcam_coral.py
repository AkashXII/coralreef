import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def get_model_device(model):
    """
    Return the device used by the model parameters.

    Examples:
        cpu
        cuda:0
    """

    for parameter in model.parameters():
        return parameter.device

    raise RuntimeError(
        "Could not determine the model device."
    )


def get_target_layer(model):

    layers = list(
        model.model.children()
    )

    if len(layers) < 2:
        raise RuntimeError(
            "Unexpected model architecture."
        )

    print("\nModel layers:")

    for i, layer in enumerate(layers):
        print(
            f"  [{i}] {layer.__class__.__name__}"
        )

    target_layer = layers[-2]
    classifier = layers[-1]

    print(
        f"\nGradCAM target layer: "
        f"{target_layer.__class__.__name__}"
    )

    print(
        f"Classifier head: "
        f"{classifier.__class__.__name__}"
    )

    return target_layer, classifier

def forward_backbone(
    model,
    x,
):

    layers = list(
        model.model.children()
    )[:-1]

    feature = x

    for layer in layers:

        feature = layer(feature)

        if isinstance(
            feature,
            (tuple, list),
        ):

            if len(feature) == 0:
                raise RuntimeError(
                    "Backbone layer returned an empty tuple/list."
                )

            feature = feature[0]

    if not torch.is_tensor(feature):

        raise RuntimeError(
            f"Backbone produced {type(feature)}, "
            "expected a torch.Tensor."
        )

    return feature



def forward_classifier(
    classifier,
    feature,
):
 
    feature = classifier.conv(
        feature
    )

    feature = classifier.pool(
        feature
    )


    feature = torch.flatten(
        feature,
        1,
    )
    feature = classifier.drop(
        feature
    )
    logits = classifier.linear(
        feature
    )

    return logits



def predict_class(
    model,
    classifier,
    tensor,
):


    with torch.no_grad():

        feature = forward_backbone(
            model,
            tensor,
        )

        logits = forward_classifier(
            classifier,
            feature,
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        pred_idx = int(
            probabilities.argmax(
                dim=1
            ).item()
        )

        confidence = float(
            probabilities[
                0,
                pred_idx
            ].item()
        )

    return pred_idx, confidence



class GradCAM:

    def __init__(
        self,
        model,
        target_layer,
        classifier,
    ):

        self.model = model
        self.target_layer = target_layer
        self.classifier = classifier

        self.activations = None
        self.gradients = None

        # Forward hook.
        self.forward_handle = (
            target_layer.register_forward_hook(
                self._save_activation
            )
        )

        # Backward hook.
        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self._save_gradient
            )
        )

    def _save_activation(
        self,
        module,
        inputs,
        output,
    ):
        if isinstance(
            output,
            (tuple, list),
        ):

            if len(output) == 0:
                raise RuntimeError(
                    "Target layer returned an empty tuple/list."
                )

            output = output[0]

        if not torch.is_tensor(output):

            raise TypeError(
                f"Target layer returned "
                f"{type(output)}, expected Tensor."
            )

        self.activations = output

 
    def _save_gradient(
        self,
        module,
        grad_input,
        grad_output,
    ):
        """
        Capture gradients flowing through the target layer.
        """

        if (
            grad_output is None
            or grad_output[0] is None
        ):

            raise RuntimeError(
                "Grad-CAM target layer received no gradient."
            )

        self.gradients = grad_output[0]

    def remove_hooks(self):

        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(
        self,
        input_tensor,
        class_idx,
    ):

        # Clear previous gradients.
        self.model.zero_grad(
            set_to_none=True
        )


        input_tensor = (
            input_tensor
            .clone()
            .detach()
        )

        input_tensor.requires_grad_(
            True
        )

        feature = forward_backbone(
            self.model,
            input_tensor,
        )

        if not feature.requires_grad:

            raise RuntimeError(
                "Backbone feature does not require gradients."
            )

        logits = forward_classifier(
            self.classifier,
            feature,
        )

        if not logits.requires_grad:

            raise RuntimeError(
                "Classification logits do not require gradients."
            )

        if logits.grad_fn is None:

            raise RuntimeError(
                "Classification logits have no grad_fn."
            )


        score = logits[
            0,
            class_idx
        ]

        score.backward()


        if self.activations is None:

            raise RuntimeError(
                "Grad-CAM activation was not captured."
            )

        if self.gradients is None:

            raise RuntimeError(
                "Grad-CAM gradient was not captured."
            )


        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            weights
            * self.activations
        ).sum(
            dim=1,
            keepdim=True,
        )


        cam = F.relu(
            cam
        )


        cam = (
            cam
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

  
        cam_min = cam.min()
        cam_max = cam.max()

        if (
            cam_max - cam_min
            < 1e-8
        ):

            cam = np.zeros_like(
                cam
            )

        else:

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        return cam


def process_image(
    image_path,
    output_path,
    model,
    classifier,
    cam_engine,
    device,
    names,
    img_size,
):
    """
    Process one image and save its Grad-CAM result.
    """

    print(
        f"\nProcessing: {image_path.name}"
    )


    img_bgr = cv2.imread(
        str(image_path)
    )

    if img_bgr is None:

        print(
            "  ERROR: Could not read image."
        )

        return False

    original_height, original_width = (
        img_bgr.shape[:2]
    )


    img_rgb = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2RGB,
    )

    img_resized = cv2.resize(
        img_rgb,
        (
            img_size,
            img_size,
        ),
    )
    tensor = torch.from_numpy(
        img_resized
    ).float()

    # HWC -> CHW
    tensor = tensor.permute(
        2,
        0,
        1,
    )

    tensor = tensor / 255.0

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.to(
        device
    )
    pred_idx, confidence = predict_class(
        model,
        classifier,
        tensor,
    )


    if isinstance(
        names,
        dict,
    ):

        pred_label = names[
            pred_idx
        ]

    elif isinstance(
        names,
        list,
    ):

        pred_label = names[
            pred_idx
        ]

    else:

        pred_label = str(
            pred_idx
        )

    print(
        f"  Prediction : "
        f"{pred_label}"
    )

    print(
        f"  Confidence : "
        f"{confidence:.4f}"
    )

    cam = cam_engine(
        tensor,
        pred_idx,
    )
    cam_resized = cv2.resize(
        cam,
        (
            original_width,
            original_height,
        ),
    )
    heatmap = cv2.applyColorMap(
        (
            cam_resized * 255
        ).astype(
            np.uint8
        ),
        cv2.COLORMAP_JET,
    )
    overlay = cv2.addWeighted(
        img_bgr,
        0.55,
        heatmap,
        0.45,
        0,
    )
    success = cv2.imwrite(
        str(output_path),
        overlay,
    )

    if not success:

        print(
            "  ERROR: Failed to save output."
        )

        return False

    print(
        f"  Saved      : "
        f"{output_path.name}"
    )

    return True

def run_batch_gradcam(
    model_path,
    test_dir,
    result_dir,
    img_size=224,
):
    model_path = Path(
        model_path
    )

    test_dir = Path(
        test_dir
    )

    result_dir = Path(
        result_dir
    )
    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    if not test_dir.exists():

        raise FileNotFoundError(
            f"Test image directory does not exist:\n"
            f"{test_dir}"
        )
    print(
        "\nLoading model..."
    )

    yolo = YOLO(
        str(model_path)
    )

    model = yolo.model

    model.eval()

    names = yolo.names
    device = get_model_device(
        model
    )

    print(
        f"Model device: "
        f"{device}"
    )

    model.to(
        device
    )
    target_layer, classifier = (
        get_target_layer(model)
    )
    cam_engine = GradCAM(
        model=model,
        target_layer=target_layer,
        classifier=classifier,
    )

    try:
        image_paths = sorted(
            [
                path
                for path in test_dir.iterdir()
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in IMAGE_EXTENSIONS
                )
            ]
        )

        if not image_paths:

            print(
                f"\nNo supported images found in:\n"
                f"{test_dir}"
            )

            return
        print(
            f"\nFound "
            f"{len(image_paths)} image(s)."
        )

        print(
            f"Input directory : "
            f"{test_dir}"
        )

        print(
            f"Output directory: "
            f"{result_dir}"
        )

        successful = 0
        failed = 0

        for image_path in image_paths:

            output_path = (
                result_dir
                / f"{image_path.stem}_gradcam.png"
            )

            try:

                success = process_image(
                    image_path=image_path,
                    output_path=output_path,
                    model=model,
                    classifier=classifier,
                    cam_engine=cam_engine,
                    device=device,
                    names=names,
                    img_size=img_size,
                )

                if success:
                    successful += 1
                else:
                    failed += 1

            except Exception as error:

                failed += 1

                print(
                    f"  ERROR: {error}"
                )

        print(
            "\n"
            + "=" * 60
        )

        print(
            "Grad-CAM batch processing complete."
        )

        print(
            f"Successful : {successful}"
        )

        print(
            f"Failed     : {failed}"
        )

        print(
            f"Results    : {result_dir}"
        )

        print(
            "=" * 60
        )
    finally:
        cam_engine.remove_hooks()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Generate Grad-CAM visualizations "
            "for all images in grad_cam/test_images."
        )
    )

    parser.add_argument(
        "--model",
        default="models/best.pt",
        help=(
            "Path to YOLO classification model. "
            "Default: models/best.pt"
        ),
    )

    parser.add_argument(
        "--test-dir",
        default="grad_cam/test_images",
        help=(
            "Directory containing test images. "
            "Default: grad_cam/test_images"
        ),
    )

    parser.add_argument(
        "--result-dir",
        default="grad_cam/result_images",
        help=(
            "Directory where Grad-CAM images are saved. "
            "Default: grad_cam/result_images"
        ),
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
        help="Model input image size. Default: 224",
    )

    args = parser.parse_args()

    run_batch_gradcam(
        model_path=args.model,
        test_dir=args.test_dir,
        result_dir=args.result_dir,
        img_size=args.imgsz,
    )