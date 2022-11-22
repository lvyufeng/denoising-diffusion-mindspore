import math
from tqdm import tqdm
import mindspore
from mindspore import nn, ops
from mindspore import ms_function, save_checkpoint, load_checkpoint, load_param_into_net
from mindspore import set_auto_parallel_context
from mindspore.communication import init, get_rank, get_group_size
from .dataset import create_dataset
from .api import value_and_grad
from .accumulator import Accumulator

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

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

        # auto mixed precision
        from .amp import DynamicLossScaler, NoLossScaler, auto_mixed_precision
        self.model = auto_mixed_precision(diffusion_model, amp_level)
        if amp_level != 'O0':
            self.loss_scaler = DynamicLossScaler(65536, 2, 1000)
        else:
            self.loss_scaler = NoLossScaler()

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

    def save(self, milestone):
        if not self.is_main_process:
            return

        data = [
            {'step': self.step},
            {'model': self.model},
            {'opt': self.opt},
            {'ema': self.ema.state_dict()},
            {'scaler': self.loss_scaler}
        ]

        save_checkpoint(data, str(self.results_folder / f'model-{milestone}.ckpt'))

    def load(self, milestone):
        data = load_checkpoint(str(self.results_folder / f'model-{milestone}.ckpt'))

        # model = self.accelerator.unwrap_model(self.model)
        # model.load_state_dict(data['model'])

        # self.step = data['step']
        # self.opt.load_state_dict(data['opt'])
        # self.ema.load_state_dict(data['ema'])

        # if exists(self.accelerator.scaler) and exists(data['scaler']):
        #     self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        from .amp import all_finite

        model = self.model
        accumulator = self.accumulator
        loss_scaler = self.loss_scaler
        grad_acc = self.gradient_accumulate_every

        def forward_fn(data, noise):
            loss = model(data, noise)
            loss = loss_scaler.scale(loss)
            return loss / grad_acc

        grad_fn = value_and_grad(forward_fn, None, self.opt.parameters)

        def train_step(data, noise):
            loss, grads = grad_fn(data, noise)
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
                loss = train_step(data, noise)
                total_loss += loss.asnumpy()
                # if accelerator.is_main_process:
                #     self.ema.to(device)
                #     self.ema.update()

                #     if self.step != 0 and self.step % self.save_and_sample_every == 0:
                #         self.ema.ema_model.eval()

                #         with torch.no_grad():
                #             milestone = self.step // self.save_and_sample_every
                #             batches = num_to_groups(self.num_samples, self.batch_size)
                #             all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                #         all_images = torch.cat(all_images_list, dim = 0)
                #         utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
                #         self.save(milestone)
                self.step += 1
                if self.step % self.gradient_accumulate_every == 0:
                    pbar.set_description(f'loss: {total_loss:.4f}')
                    pbar.update(1)
                    total_loss = 0.
                if self.step >= self.gradient_accumulate_every * self.train_num_steps:
                    break

        # accelerator.print('training complete')
