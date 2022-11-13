from mindspore import nn, ops, Tensor
from mindspore.ops import constexpr
from mindspore.ops.operations.image_ops import ResizeBilinearV2, ResizeLinear1D
from mindspore.ops._primitive_cache import _get_cache_prim


@constexpr
def _check_scale_factor(shape, scale_factor):
    if isinstance(scale_factor, tuple) and len(scale_factor) != len(shape[2:]):
        raise ValueError(f"the number of 'scale_fator' must match to inputs.shape[2:], "
                         f"but get scale_factor={scale_factor}, inputs.shape[2:]={shape[2:]}")


def _interpolate_output_shape(shape, scales, sizes, mode):
    """calculate output shape"""
    if sizes is not None:
        if mode == "nearest":
            return sizes
        return Tensor(sizes)

    ret = ()        
    for i in range(len(shape[2:])):
        if isinstance(scales, float):
            out_i = int(scales * shape[i+2])
        else:
            out_i = int(scales[i] * shape[i+2])
        ret = ret + (out_i,)
    if mode == "nearest":
        return ret
    return Tensor(ret)


class Upsample(nn.Cell):
    def __init__(self, size = None, scale_factor = None,
                 mode: str = 'nearest', align_corners = False):
        super().__init__()
        if mode not in ['nearest', 'linear', 'bilinear']:
            raise ValueError(f'do not support mode :{mode}.')
        if size and scale_factor:
            raise ValueError(f"can not set 'size' and 'scale_fator' at the same time.")
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def construct(self, inputs):
        inputs_shape = inputs.shape
        _check_scale_factor(inputs_shape, self.scale_factor)
        sizes = _interpolate_output_shape(inputs_shape, self.scale_factor, self.size, self.mode)
        if self.mode == 'nearest':
            interpolate = _get_cache_prim(ops.ResizeNearestNeighbor)(sizes, self.align_corners)
            return interpolate(inputs)
        elif self.mode == 'linear':
            interpolate = _get_cache_prim(ResizeLinear1D)('align_corners' if self.align_corners else 'half_pixel')
            return interpolate(inputs, sizes)
        elif self.mode == 'bilinear':
            interpolate = _get_cache_prim(ResizeBilinearV2)(self.align_corners, True if self.align_corners==False else False)
            return interpolate(inputs, sizes)
        return 