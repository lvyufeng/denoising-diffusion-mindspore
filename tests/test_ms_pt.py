from denoising_diffusion_pytorch import Unet as ptUnet
from denoising_diffusion_pytorch import GaussianDiffusion as ptGaussianDiffusion
from ddm import Unet as msUnet
from ddm import GaussianDiffusion as msGaussianDiffusion
from mindspore import Tensor, Parameter, load_param_into_net
import numpy as np
import torch
import mindspore

mindspore.set_context(mode=mindspore.PYNATIVE_MODE)

def test_trainer():
    ms_model = msUnet(
        dim = 64,
        dim_mults = (1, 2, 4, 8)
    )
    ms_diffusion = msGaussianDiffusion(
        ms_model,
        image_size = 128,
        timesteps = 1000,   # number of steps
        loss_type = 'l1'    # L1 or L2
    )
    pt_model = ptUnet(
        dim = 64,
        dim_mults = (1, 2, 4, 8)
    )
    pt_diffusion = ptGaussianDiffusion(
        pt_model,
        image_size = 128,
        timesteps = 1000,   # number of steps
        loss_type = 'l1'    # L1 or L2
    )
    for name, param in ms_diffusion.parameters_and_names():
        print('ms', name, param.shape)

    ms_ckpt = {}
    for name, param in pt_diffusion.state_dict().items():
        if 'norm' in name:
            name = name.replace('weight', 'gamma')
            name = name.replace('bias', 'beta')
        print('pt', name, param.shape)
        ms_ckpt[name] = Parameter(Tensor(param.numpy()), name)
    
    unloaded = load_param_into_net(ms_diffusion, ms_ckpt)
    print(unloaded)

    training_images = np.random.rand(8, 3, 128, 128).astype(np.float32)
    t = np.random.randint(0, 1000, (8,))

    pt_data = torch.tensor(training_images)
    ms_data = Tensor(training_images, mindspore.float32)
    pt_t = torch.tensor(t)
    ms_t = Tensor(t, mindspore.int32)
    pt_out = pt_diffusion(pt_data, pt_t)
    ms_out = ms_diffusion(ms_data, ms_t)
    # pt_out = pt_model(pt_data, pt_t)
    # ms_out = ms_model(ms_data, ms_t)
    print(pt_out, ms_out)
    assert np.allclose(pt_out.detach().numpy(), ms_out.asnumpy(), 1e-3, 1e-3)