from download import download
from ddm import Unet, GaussianDiffusion, Trainer

url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
path = download(url, './102flowers', 'tar.gz')

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    self_condition = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,             # number of steps
    loss_type = 'l1',            # L1 or L2
    beta_schedule = 'cosine',
    p2_loss_weight_gamma = 1.0,
    p2_loss_weight_k = 1,
)

trainer = Trainer(
    diffusion,
    path,
    train_batch_size = 16,
    train_lr = 2e-4,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp_level = 'O1',                        # turn on mixed precision
    save_and_sample_every = 1000,
    num_samples = 16
)

trainer.train()
