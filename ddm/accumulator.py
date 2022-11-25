import mindspore
from mindspore import ms_class, Tensor, Parameter, ops

@ms_class
class Accumulator():
    def __init__(self, optimizer, accumulate_step, total_step=None, clip_norm=1.0):
        # super().__init__()
        self.optimizer = optimizer
        self.clip_norm = clip_norm
        self.inner_grads = optimizer.parameters.clone(prefix="accumulate_", init='zeros')
        self.zeros = optimizer.parameters.clone(prefix="zeros_", init='zeros')
        self.counter = Parameter(Tensor(1, mindspore.int32), 'counter_')
        assert accumulate_step > 0
        self.accumulate_step = accumulate_step
        if total_step is not None:
            assert total_step > accumulate_step and total_step > 0
        self.total_step = total_step
        self.map = ops.Map()
        self.partial = ops.Partial()
    
    def __call__(self, grads):
        if self.counter % self.accumulate_step == 0:
            clip_grads = ops.clip_by_global_norm(self.inner_grads, self.clip_norm)
            # clip_grads, _ = clip_grad_norm(self.inner_grads, self.clip_norm)
            self.optimizer(clip_grads)
            success = self.map(self.partial(ops.assign), self.inner_grads, self.zeros)
        else:
            success = self.map(self.partial(ops.assign_add), self.inner_grads, grads)
        # for last step which can not be divided by accumulate_step
        if self.total_step is not None and self.counter == self.total_step:
            clip_grads = ops.clip_by_global_norm(self.inner_grads, self.clip_norm)
            # clip_grads, _ = clip_grad_norm(self.inner_grads, self.clip_norm)
            self.optimizer(clip_grads)
        ops.assign_add(self.counter, Tensor(1, mindspore.int32))

        return success
