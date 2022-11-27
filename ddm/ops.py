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
    if not isinstance(axis, int):
        type_axis = type(axis).__name__
        raise TypeError(f" the type of 'axis' must be 'int', but got '{axis}' with type '{type_axis}'.")
    softmax_ = _get_cache_prim(ops.Softmax)(axis=axis)
    return softmax_(x)

def bhdi_bhdj_bhij_old(x, y):
    if gpu_target:
        einsum = _get_cache_prim(ops.Einsum)('b h d i, b h d j -> b h i j')
        return einsum((x, y))
    else:
        return (x.expand_dims(-1) * y.expand_dims(-2)).sum(2)

def bhdi_bhdj_bhij(x, y):
    bmm = _get_cache_prim(ops.BatchMatMul)()
    return bmm(x.swapaxes(2, 3), y)

def bhij_bhdj_bhid_old(x, y):
    if gpu_target:
        einsum = _get_cache_prim(ops.Einsum)('b h i j, b h d j -> b h i d')
        return einsum((x, y))
    else:
        return (x.expand_dims(3) * y.expand_dims(2)).sum(-1)

def bhij_bhdj_bhid(x, y):
    bmm = _get_cache_prim(ops.BatchMatMul)()
    return bmm(x, y.swapaxes(2, 3))