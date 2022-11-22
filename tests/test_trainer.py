from download import download
from ddm import Unet, GaussianDiffusion, Trainer
from mindspore import Profiler

def test_trainer():
    url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
    path = download(url, './102flowers', 'tar.gz')

    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8)
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = 128,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type = 'l1'            # L1 or L2
    )

    trainer = Trainer(
        diffusion,
        path,
        train_batch_size = 16,
        train_lr = 8e-5,
        train_num_steps = 70000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp_level= 'O1',                        # turn on mixed precision
    )

    # profiler = Profiler()
    trainer.train()
    # profiler.analyse()