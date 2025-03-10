import torch


def main():
    print("PyTorch version: ", torch.__version__)
    print("CUDA available: ", torch.cuda.is_available())
    print("CUDA version: ", torch.version.cuda)
    print("cuDNN version: ", torch.backends.cudnn.version())
    print("CUDA device count: ", torch.cuda.device_count())
    print("CUDA device name: ", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()
