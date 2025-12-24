# CUDA Image Convolution

CUDA sample that applies 3x3 image filters (identity, Gaussian blur, edge detect, sharpen) using shared-memory tiling with halo exchange. A small Python CLI wraps build and execution for quick demos.

## Features
- 16x16 thread blocks cooperatively load an 18x18 shared tile (block + 1px halo) to avoid redundant global reads.
- Boundary sampling is clamped in the load step so the inner convolution loop stays branch-free.
- Filter coefficients live in constant memory for fast broadcast to the SM.
- Python CLI (`scripts/cli.py`) handles building via `make`, piping menu input to the binary, and optional visualization.

## Prerequisites
- CUDA toolkit (`nvcc`) and an NVIDIA GPU (set `ARCH` to your target SM, default `sm_75`).
- `make` and `wget` for fetching STB headers.
- Python 3.8+; `matplotlib` only if you want `--show` visualization.

## Build
```bash
# Build binary to bin/convolution
make ARCH=sm_75
```

## Run via CLI
```bash
# Build (if needed) and run a Gaussian blur, then display results
python scripts/cli.py --build --image data/input.png --filter gaussian --show
```

Filters: `identity`, `gaussian`, `edge`, `sharpen`.

The CLI pipes the expected menu commands to the C++ app: set file, set filter, process, exit. Output is written to `output.png` in the repo root.

## Algorithm: Shared Tile + Halo
- Each block owns a 16x16 output region; shared memory allocates `(16 + 2*1)^2` RGB floats (18x18) to hold block data plus a 1-pixel halo.
- Threads load their center pixels; boundary threads also load left/right/top/bottom halos, and four corner threads load the diagonal halo cells.
- Global reads clamp coordinates to `[0, width-1]` / `[0, height-1]`, so the convolution loop has no boundary condition branches.
- After a `__syncthreads()`, each thread multiplies the 3x3 neighborhood from shared memory by the constant-memory kernel and writes a clamped 0–255 result.

## Repository Layout
```
├── Makefile
├── README.md
├── .gitignore
├── include/           # STB headers downloaded by make
├── src/
│   ├── kernels.cu     # CUDA kernel with shared-memory tiling + halo
│   └── main.cpp       # Host driver, launches kernel
├── scripts/
│   └── cli.py         # Python CLI: build/run/show
└── data/              # Put your input images here
```

## Notes
- If you change the GPU target, pass `ARCH=sm_xy` to `make` or set it in the Makefile.
- The build downloads `stb_image.h` and `stb_image_write.h` into `include/`; they are git-ignored.
