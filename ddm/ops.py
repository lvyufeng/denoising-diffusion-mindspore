import mindspore
from mindspore import ops, Tensor, context
from mindspore.ops._primitive_cache import _get_cache_prim

gpu_target = (context.get_context("device_target") == "GPU")

def rsqrt(x):
    rsqrt_op = _get_cache_prim(ops.Rsqrt)()
    return rsqrt_op(x)

def rearrange(head, inputs):
    b, hc, x, y = inputs.shape
    c = hc // head
    return inputs.reshape((b, head, c, x*y))

def randint(low, high, size, dtype=mindspore.int32):
    uniform_int = _get_cache_prim(ops.UniformInt)()
    return uniform_int(size, Tensor(low, mindspore.int32), Tensor(high, mindspore.int32)).astype(dtype)

def random():
    uniform = _get_cache_prim(ops.UniformReal)()
    return uniform((1,))

def randn_like(x, dtype=None):
    if dtype is None:
        dtype = x.dtype
    normal = _get_cache_prim(ops.StandardNormal)()
    return normal(x.shape).astype(dtype)

def randn(shape, dtype=None):
    if dtype is None:
        dtype = mindspore.float32
    normal = _get_cache_prim(ops.StandardNormal)()
    return normal(shape).astype(dtype)

def cumprod(input, dim, dtype=None):
    cumprod_op = _get_cache_prim(ops.CumProd)()
    output = cumprod_op(input, dim)
    if dtype:
        output = _get_cache_prim(ops.Cast)()(output, dtype)
    return output

def softmax(x, axis=-1):
    if gpu_target:
        softmax_ = _get_cache_prim(ops.Softmax)(axis=axis)
        return softmax_(x)
    exp_ = _get_cache_prim(ops.Exp)()
    reduce_sum_ = _get_cache_prim(ops.ReduceSum)(True)

    x_max = x.max(axis=axis, keepdims=True)
    x_exp = exp_(x - x_max)
    partion = reduce_sum_(x_exp, axis)
    return x_exp / partion
