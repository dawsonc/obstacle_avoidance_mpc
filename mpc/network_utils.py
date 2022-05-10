"""Utilities for exporting neural networks"""
import torch


def pytorch_to_nnet(
    module: torch.nn.Module, input_size: int, output_size: int, save_path: str
):
    """Save the provided pytorch module to the NNet format"""
    dummy_input = torch.zeros(input_size)
    # input_names = [f"state_{i}" for i in range(input_size)]
    # output_names = [f"control_{i}" for i in range(output_size)]
    torch.onnx.export(
        module,
        dummy_input,
        save_path,
        verbose=True,
        # input_names=input_names,
        # output_names=output_names,
    )
