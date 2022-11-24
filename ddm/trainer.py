import math
from tqdm import tqdm
from pathlib import Path
import mindspore
from mindspore import nn, ops
from mindspore import ms_function, save_checkpoint, load_checkpoint, load_param_into_net
from mindspore import set_auto_parallel_context
from mindspore.communication import init, get_rank, get_group_size
from mindspore.parallel._utils import _get_device_num, _get_gradients_mean

from .dataset import create_dataset
from .api import value_and_grad
from .accumulator import Accumulator
from .utils import to_image

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp_level = 'O1',
        jit = True,
        akg = True,
        distributed = False,
    ):
        super().__init__()
        if jit and akg:
            mindspore.set_context(enable_graph_kernel=True)
        # distributed training
        self.distributed = distributed
        if distributed:
            init()
            rank_id = get_rank()
            rank_size = get_group_size()
            set_auto_parallel_context(parallel_mode=mindspore.ParallelMode.DATA_PARALLEL,
                                      gradients_mean=True)
        else:
            rank_id = 0
            rank_size = 1
        
        self.is_main_process = True if rank_id == 0 else False
        if self.is_main_process:
            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok = True)

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader
        self.ds = create_dataset(folder, self.image_size, augment_horizontal_flip=augment_horizontal_flip, \
            batch_size=train_batch_size, num_shards=rank_size, shard_id=rank_id, shuffle=True, drop_remainder=True)
        dataset_size = self.ds.get_dataset_size()
        self.ds = self.ds.repeat(int(train_num_steps * gradient_accumulate_every // dataset_size) + 1)
        # optimizer
        self.opt = nn.Adam(diffusion_model.trainable_params(), train_lr, adam_betas[0], adam_betas[1])

        # accumulator
        self.gradient_accumulate_every = gradient_accumulate_every
        self.accumulator = Accumulator(self.opt, gradient_accumulate_every)

        # for logging results in a folder periodically

        # step counter state
        self.step = 0
        self.results_folder = results_folder
        self.jit = jit
        self.model = diffusion_model
        self.amp_level = amp_level

    def save(self, milestone):
        if not self.is_main_process:
            return

        append_dict = self.opt.parameters_dict()
        append_dict['step'] = self.step

        save_checkpoint(self.model, str(self.results_folder + f'/model-{milestone}.ckpt'),
                        append_dict=append_dict)

    def load(self, milestone):
        data = load_checkpoint(str(self.results_folder + f'/model-{milestone}.ckpt'))

        # model = self.accelerator.unwrap_model(self.model)
        # model.load_state_dict(data['model'])

        # self.step = data['step']
        # self.opt.load_state_dict(data['opt'])
        # self.ema.load_state_dict(data['ema'])

        # if exists(self.accelerator.scaler) and exists(data['scaler']):
        #     self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        model = self.model
        accumulator = self.accumulator
        loss_scaler = self.loss_scaler
        grad_acc = self.gradient_accumulate_every

        # auto mixed precision
        from .amp import DynamicLossScaler, NoLossScaler, auto_mixed_precision, all_finite
        model = auto_mixed_precision(model, self.amp_level)
        if self.amp_level != 'O0':
            self.loss_scaler = DynamicLossScaler(65536, 2, 1000)
        else:
            self.loss_scaler = NoLossScaler()

        if self.distributed:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            grad_reducer = nn.DistributedGradReducer(self.opt.parameters, mean, degree)
        else:
            grad_reducer = ops.identity

        def forward_fn(data, noise):
            loss = model(data, noise)
            loss = loss_scaler.scale(loss)
            return loss / grad_acc

        grad_fn = value_and_grad(forward_fn, None, self.opt.parameters)

        def train_step(data, noise):
            loss, grads = grad_fn(data, noise)
            grads = grad_reducer(grads)
            status = all_finite(grads)
            if status:
                loss = loss_scaler.unscale(loss)
                grads = loss_scaler.unscale(grads)
                loss = ops.depend(loss, accumulator(grads))
            loss_scaler.adjust(status)
            return loss

        if self.jit:
            train_step = ms_function(train_step)

        data_iterator = self.ds.create_tuple_iterator()
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not self.is_main_process) as pbar:
            total_loss = 0.
            for (data, noise) in data_iterator:
                model.set_train()
                loss = train_step(data, noise)
                total_loss += float(loss.asnumpy())

                self.step += 1
                if self.step % self.gradient_accumulate_every == 0:
                    pbar.set_description(f'loss: {total_loss:.4f}')
                    pbar.update(1)
                    total_loss = 0.

                if self.is_main_process:
                #     self.ema.to(device)
                #     self.ema.update()
                    accumulate_step = self.step // self.gradient_accumulate_every
                    accumulate_remain_step = self.step % self.gradient_accumulate_every
                    if accumulate_step != 0 and \
                        accumulate_step % self.save_and_sample_every == 0 and \
                        accumulate_remain_step == (self.gradient_accumulate_every - 1):
                        self.save(accumulate_step)

                        model.set_train(False)

                        batches = num_to_groups(self.num_samples, self.batch_size)
                        print(batches)
                        all_images_list = list(map(lambda n: model.sample(batch_size=n), batches))

                        all_images = ops.concat(all_images_list, axis = 0)
                        to_image(all_images, str(self.results_folder + f'/sample-{accumulate_step}.png'), nrow = int(math.sqrt(self.num_samples)))

                if self.step >= self.gradient_accumulate_every * self.train_num_steps:
                    break

        print('training complete')
