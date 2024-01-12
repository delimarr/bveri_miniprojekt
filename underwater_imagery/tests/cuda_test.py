import torch


def test_cuda_available() -> None:
    assert torch.cuda.is_available() == True


if __name__ == "__main__":
    test_cuda_available()
