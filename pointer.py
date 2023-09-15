# Copyright 2023 ⓒ Daemyung Jang.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import triton
import triton.language as tl


@triton.jit
def block_pointer(
    y_ptr: tl.tensor,
    x_ptr: tl.tensor,
    y_size: tl.int32,
    x_size: tl.int32,
    y_block_size: tl.constexpr,
    x_block_size: tl.constexpr,
):
    y_block_ptr = tl.make_block_ptr(
        y_ptr,
        shape=(y_size, x_size),
        strides=(x_size, 1),
        offsets=(0, 0),
        block_shape=(y_block_size, x_block_size),
        order=(1, 0),
    )
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(y_size, x_size),
        strides=(x_size, 1),
        offsets=(0, 0),
        block_shape=(y_block_size, x_block_size),
        order=(1, 0),
    )

    x = tl.load(x_block_ptr)
    tl.store(y_block_ptr, x)


@triton.jit
def pointer_block(
    y_ptr: tl.tensor,
    x_ptr: tl.tensor,
    y_size: tl.int32,
    x_size: tl.int32,
    y_block_size: tl.constexpr,
    x_block_size: tl.constexpr,
):
    y_offsets = tl.arange(0, y_block_size)
    x_offsets = tl.arange(0, x_block_size)
    offsets = y_offsets[:, None] * x_size + x_offsets[None, :]
    x = tl.load(x_ptr + offsets)
    tl.store(y_ptr + offsets, x)


def dispatch(kernel: triton.jit, y: torch.Tensor, x: torch.Tensor):
    y_size, x_size = x.shape
    kernel[(1,)](y, x, y_size, x_size, triton.next_power_of_2(y_size), triton.next_power_of_2(x_size))


def verify_result():
    factory_kwargs = {"device": "cuda", "dtype": torch.float32}
    x = torch.rand(16, 16, **factory_kwargs)
    a = torch.empty_like(x)
    b = torch.empty_like(x)
    dispatch(block_pointer, a, x)
    dispatch(pointer_block, b, x)
    torch.allclose(a, b)


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["y_size", "x_size"],
            x_vals=[2**i for i in range(5, 9, 1)],
            line_arg="backend",
            line_vals=["block pointer", "pointer block"],
            line_names=["block pointer", "pointer block"],
            ylabel="milliseconds",
            plot_name="pointer",
            args={"dtype": torch.float32},
        )
    ]
)
def benchmark(y_size, x_size, dtype, backend):
    factory_kwargs = {"device": "cuda", "dtype": dtype}
    x = torch.rand(y_size, x_size, **factory_kwargs)
    y = torch.empty_like(x)

    if backend == "block pointer":
        return triton.testing.do_bench_cudagraph(lambda: dispatch(block_pointer, y, x))
    else:
        return triton.testing.do_bench_cudagraph(lambda: dispatch(pointer_block, y, x))


def main():
    torch.cuda.set_stream(torch.cuda.Stream())
    verify_result()
    benchmark.run(show_plots=True, print_data=True)


if __name__ == "__main__":
    main()
