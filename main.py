from models.model import Autoencoder

from models.optimized_model import Autoencoder_Optimized
from models.optimized_model_wide import Autoencoder_Optimized_Wide

from models.interpolation_models.model_bilinear import Autoencoder_Bilinear
from models.interpolation_models.model_convTranspose import Autoencoder_convTranspose

from models.activation_models.model_leaky_relu import Autoencoder_LeakyReLU
from models.activation_models.model_sigmoid import Autoencoder_Sigmoid

from models.deep_models.model_1 import Autoencoder_Deep_1
from models.deep_models.model_3 import Autoencoder_Deep_3
from models.deep_models.model_4 import Autoencoder_Deep_4
from models.deep_models.model_5 import Autoencoder_Deep_5
from models.deep_models.model_6 import Autoencoder_Deep_6
from models.deep_models.model_7 import Autoencoder_Deep_7

from models.wide_models.model_64 import Autoencoder_Wide_64
from models.wide_models.model_128 import Autoencoder_Wide_128
from models.wide_models.model_512 import Autoencoder_Wide_512

from trainer.mse_loss import Trainer_MSE
from trainer.vgg19_loss import Trainer_perceputal

from validation_func.tester import Tester
from validation_func.image_metric import Benchmark


import torch


def check_device():
    if torch.cuda.is_available():
        print("GPU is available and being used")
        return torch.device("cuda")
    else:
        print("GPU is not available, using CPU instead")
        return torch.device("cpu")


def train_base_model(device, dataset_path):
    ae = Autoencoder()
    ae.to(device)
    ae.load_state_dict(torch.load("trained_models/base_model/060"))
    ae.print_model()
    print("Base Model")
    train_path = "trained_models/base_model/"
    trainer = Trainer_MSE(ae, device, train_path, dataset_path)
    trainer.train(num_epochs=20)


def train_optimized_model(device, dataset_path):
    ae = Autoencoder_Optimized()
    ae.to(device)
    ae.print_model()
    print("Optimized Model")
    train_path = "trained_models/optimized_model/perceptual_reg"
    trainer = Trainer_perceputal(
        ae, device, 0.2, train_path, dataset_path)
    trainer.train(num_epochs=80)


def train_interpolation_models(device, dataset_path):
    aes = [
        {
            "ae": Autoencoder_Bilinear(),
            "name": "Autoencoder_Bilinear",
            "train_path": "trained_models/interpolation_models/bilinear/"
        },
        {
            "ae": Autoencoder_convTranspose(),
            "name": "Autoencoder_convTranspose",
            "train_path": "trained_models/interpolation_models/convTranspose/"
        }
    ]
    for ae in aes:
        ae["ae"].to(device)
        ae["ae"].print_model()
        print(ae["name"])
        trainer = Trainer_MSE(ae["ae"], device, ae["train_path"], dataset_path)
        trainer.train(num_epochs=60)


def train_activation_models(device, dataset_path):
    aes = [
        {
            "ae": Autoencoder_LeakyReLU(),
            "name": "Autoencoder_LeakyReLU",
            "train_path": "trained_models/activation_models/leakyReLU/"
        },
        {
            "ae": Autoencoder_Sigmoid(),
            "name": "Autoencoder_Sigmoid",
            "train_path": "trained_models/activation_models/sigmoid/"
        }
    ]
    for ae in aes:
        ae["ae"].to(device)
        ae["ae"].print_model()
        print(ae["name"])
        trainer = Trainer_MSE(ae["ae"], device, ae["train_path"], dataset_path)
        trainer.train(num_epochs=60)


def train_deep_models(device, dataset_path):
    aes = [
        {
            "ae": Autoencoder_Deep_1(),
            "name": "Autoencoder_Deep_1",
            "train_path": "trained_models/deep_models/deep_1/"
        },
        {
            "ae": Autoencoder_Deep_3(),
            "name": "Autoencoder_Deep_3",
            "train_path": "trained_models/deep_models/deep_3/"
        },
        {
            "ae": Autoencoder_Deep_4(),
            "name": "Autoencoder_Deep_4",
            "train_path": "trained_models/deep_models/deep_4/"
        },
        {
            "ae": Autoencoder_Deep_5(),
            "name": "Autoencoder_Deep_5",
            "train_path": "trained_models/deep_models/deep_5/"
        },
        {
            "ae": Autoencoder_Deep_6(),
            "name": "Autoencoder_Deep_6",
            "train_path": "trained_models/deep_models/deep_6/"
        },
        {
            "ae": Autoencoder_Deep_7(),
            "name": "Autoencoder_Deep_7",
            "train_path": "trained_models/deep_models/deep_7/"
        },
    ]
    for ae in aes:
        ae["ae"].to(device)
        ae["ae"].print_model()
        print(ae["name"])
        trainer = Trainer_MSE(ae["ae"], device, ae["train_path"], dataset_path)
        trainer.train(num_epochs=60)


def train_wide_models(device, dataset_path):
    aes = [
        {
            "ae": Autoencoder_Wide_64(),
            "name": "Autoencoder_Wide_64",
            "train_path": "trained_models/wide_models/wide_64/"
        },
        {
            "ae": Autoencoder_Wide_128(),
            "name": "Autoencoder_Wide_128",
            "train_path": "trained_models/wide_models/wide_128/"
        },
        {
            "ae": Autoencoder_Wide_512(),
            "name": "Autoencoder_Wide_512",
            "train_path": "trained_models/wide_models/wide_512/"
        }
    ]
    for ae in aes:
        ae["ae"].to(device)
        ae["ae"].print_model()
        print(ae["name"])
        trainer = Trainer_MSE(ae["ae"], device, ae["train_path"], dataset_path)
        trainer.train(num_epochs=60)


def train_perceptual_loss(device, dataset_path):
    aes = [
        {
            "ae": Autoencoder(),
            "name": "Autoencoder_30",
            "train_path": "trained_models/loss_models/perceptual_reg/30/",
            "perceptual_loss": 0.3
        },
        {
            "ae": Autoencoder(),
            "name": "Autoencoder_40",
            "train_path": "trained_models/loss_models/perceptual_reg/40/",
            "perceptual_loss": 0.4
        },
    ]
    for ae in aes:
        ae["ae"].to(device)
        ae["ae"].print_model()
        print(ae["name"])
        trainer = Trainer_perceputal(
            ae["ae"], device,  ae["perceptual_loss"], ae["train_path"], dataset_path)
        trainer.train(num_epochs=60)


def benchmark(device, val_dataset):
    aes = [
        {
            "ae": Autoencoder(),
            "name": "Autoencoder",
            "train_path": "trained_models/base_model/",
            "psnr_path": "np_arr/base_psnr_80.dat",
            "ssim_path": "np_arr/base_ssim_80.dat"
        },
        {
            "ae": Autoencoder_Optimized(),
            "name": "Autoencoder_Optimized perceptual",
            "train_path": "trained_models/optimized_model/perceptual_reg",
            "psnr_path": "np_arr/optimized_psnr_reg.dat",
            "ssim_path": "np_arr/optimized_ssim_reg.dat"
        },
        {
            "ae": Autoencoder_Optimized(),
            "name": "Autoencoder_Optimized mse",
            "train_path": "trained_models/optimized_model/mse_loss",
            "psnr_path": "np_arr/optimized_psnr_mse.dat",
            "ssim_path": "np_arr/optimized_ssim_mse.dat"
        }
    ]
    b = Benchmark(aes, device, val_dataset)
    b.run()
    b.plot_psnr()
    b.plot_ssim()


def main():
    device = check_device()
    train_dataset_path = "dataset/train"
    val_dataset = "dataset/validation"
    # val_dataset = "dataset/ImageNet"
    # val_dataset = "dataset/OriSet/"
    # val_dataset = "dataset/TextSet/"

    mode = 1
    if mode == 0:
        print("Training Mode")
        train_base_model(device, train_dataset_path)
        train_optimized_model(device, train_dataset_path)
        train_interpolation_models(device, train_dataset_path)
        train_activation_models(device, train_dataset_path)
        train_deep_models(device, train_dataset_path)
        train_wide_models(device, train_dataset_path)
        train_perceptual_loss(device, train_dataset_path)
    elif mode == 1:
        print("Output Mode")
        model = Autoencoder_Optimized()
        model.load_state_dict(torch.load(
            "trained_models/optimized_model/mse_loss/075"))
        model.eval()
        test = Tester(model, (128, 128),
                      "test_img.jpg", "")
        test.show_super_resolution_image()
    else:
        print("Benchmark Mode")
        benchmark(device, val_dataset)


if __name__ == "__main__":
    main()
