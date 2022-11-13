import mindspore
from mindspore import ops, Tensor
from mindspore.ops._primitive_cache import _get_cache_prim

def rsqrt(x):
    rsqrt_op = _get_cache_prim(ops.Rsqrt)()
    return rsqrt_op(x)

def rearrange(head, inputs):
    b, hc, x, y = inputs.shape
    c = hc // head
    return inputs.reshape((b, head, c, x*y))

def randint(low, high, size, dtype=mindspore.int32):
    uniform_int = _get_cache_prim(ops.UniformInt)()
    return uniform_int(Tensor(high), Tensor(low), size)

def random():
    uniform = _get_cache_prim(ops.UniformReal)()
    return uniform(())

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