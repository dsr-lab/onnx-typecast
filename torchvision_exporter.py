import torch
import torchvision.models as models


def run_onnx_exporter():
    model = models.mobilenet_v3_small(pretrained=True).features
    model.eval()

    onnx_program = torch.onnx.export(
        model,
        args=(),
        kwargs={"input": torch.randn(1, 3, 256, 256)},
        dynamo=True,
    )
    onnx_program.save("features_extractor.onnx")

    print("ONNX export successful!")


if __name__ == "__main__":

    run_onnx_exporter()
