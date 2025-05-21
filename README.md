# GPU-Accelerated Python Calculator

An interactive command-line calculator that leverages Python's powerful scientific libraries (NumPy, SciPy) for CPU-based calculations and CuPy for GPU acceleration (NVIDIA CUDA & AMD ROCm). It provides a flexible environment for performing a wide range of mathematical operations, from basic arithmetic to complex linear algebra, FFTs, and special functions.

## Features

*   **Interactive CLI:** Easy-to-use command-line interface.
*   **GPU Acceleration:** Utilizes CuPy to accelerate computations on compatible NVIDIA (CUDA) or AMD (ROCm/HIP) GPUs.
*   **CPU Fallback:** Seamlessly falls back to NumPy/SciPy for CPU-based calculations if a GPU/CuPy is not available or if CPU mode is selected.
*   **Dynamic Namespace:** Access functions from `numpy` (as `np` or `xp`), `cupy` (as `cp` or `xp`), `scipy` submodules (`linalg`, `fft`, `special`, `stats`, `constants`), and Python's `math`/`cmath`.
*   **`xp` Alias:** Use `xp` as a convenient alias for the currently active array library (`cp` for GPU, `np` for CPU).
*   **Rich Functionality:**
    *   Basic arithmetic and mathematical functions (trigonometry, logarithms, exponentials, etc.).
    *   Array creation and manipulation.
    *   Linear algebra (determinants, inverses, eigenvalues, SVD, etc.).
    *   Fast Fourier Transforms (FFT).
    *   Special mathematical functions (gamma, beta, error functions, etc.).
    *   Random number generation.
    *   Basic statistical functions and distributions (from SciPy).
    *   Physical constants (from SciPy).
*   **Safe Evaluation:** Uses a restricted `eval()` environment for security.
*   **Performance Timing:** Displays the time taken for each calculation.
*   **Informative Help:** Built-in `help` command lists available functions and usage.

## Requirements

*   **Python 3.7+**
*   **NumPy** (Essential for all operations)
*   **SciPy** (Highly recommended for advanced CPU functions, stats, constants)
*   **CuPy** (Optional, for GPU acceleration)
    *   **For NVIDIA GPUs:** A compatible NVIDIA GPU with CUDA Toolkit installed.
    *   **For AMD GPUs:** A compatible AMD GPU with ROCm SDK installed.

## Installation

1.  **Clone the repository or download `gpu-calc.py`**.

2.  **Install Python:** If you don't have Python 3.7 or newer, download and install it from [python.org](https://www.python.org/downloads/). Make sure `pip` is included and in your system's PATH.

3.  **Install Core Libraries (NumPy & SciPy for CPU mode):**
    Open your terminal or command prompt and run:
    ```bash
    pip install numpy scipy
    ```

4.  **Install CuPy for GPU Acceleration (Optional):**

    *   **Important:** CuPy installation is specific to your GPU hardware (NVIDIA or AMD) and the installed driver/toolkit versions (CUDA or ROCm).
    *   **Please refer to the official CuPy Installation Guide for the correct command for your system:** [https://docs.cupy.dev/en/stable/install.html](https://docs.cupy.dev/en/stable/install.html)

    **For NVIDIA GPUs (CUDA):**
    1.  Ensure you have a compatible NVIDIA driver and the CUDA Toolkit installed. You can find your CUDA version by running `nvcc --version` (if CUDA is installed) or checking your NVIDIA driver documentation.
    2.  Install the appropriate CuPy package. For example, if you have CUDA 11.8:
        ```bash
        pip install cupy-cuda11x  # This usually installs the latest for CUDA 11.x series
        # Or be more specific, e.g., pip install cupy-cuda118
        ```
        Replace `11x` or `118` with your CUDA version (e.g., `cupy-cuda12x` for CUDA 12.x).

    **For AMD GPUs (ROCm):**
    1.  Ensure you have a compatible AMD driver and the ROCm SDK installed.
    2.  Install the appropriate CuPy package. For example, if you have ROCm 5.7:
        ```bash
        pip install cupy-rocm-5-7
        ```
        Replace `5-7` with your ROCm version (e.g., `cupy-rocm-6-0` for ROCm 6.0).

    **Verify CuPy Installation (Optional):**
    After installing CuPy, you can try importing it in Python to verify:
    ```python
    import cupy
    print(cupy.show_config())
    ```

## How to Run

Navigate to the directory containing `gpu-calc.py` in your terminal and run:

```bash
python gpu-calc.py
```

You will be greeted by the calculator's prompt.

## Usage

The calculator starts in GPU mode if CuPy is available and configured, otherwise it defaults to CPU mode.

### Commands

*   `help`: Displays the guide to available functions and modules.
*   `mode gpu`: Switches to GPU mode (uses CuPy if available).
*   `mode cpu`: Switches to CPU mode (uses NumPy/SciPy).
*   `exit` or `quit`: Exits the calculator.

### Entering Expressions

Enter mathematical expressions using Python syntax.

**The `xp` Alias:**
The most convenient way to access array functions is by prefixing them with `xp.`.
*   In GPU mode, `xp` refers to `cupy`.
*   In CPU mode, `xp` refers to `numpy`.

This allows you to write expressions that work seamlessly in both modes.

**Accessing Modules:**
*   **NumPy:** `np.` (always available, `xp` points to it in CPU mode)
*   **CuPy:** `cp.` (available if installed, `xp` points to it in GPU mode)
*   **SciPy Submodules:** `linalg.`, `fft.`, `special.`, `stats.`, `constants.`
    *   In GPU mode, these attempt to use `cupyx.scipy` components first.
    *   In CPU mode, or if `cupyx.scipy` components are missing, these use `scipy` modules.
*   **Python Math:** `math.` (for scalar math), `cmath.` (for scalar complex math)

### Examples

```
calc (CuPy(CUDA)/CuPyX.SciPy)> xp.array([1, 2, 3]) * 2
Result:
[2 4 6]
(Calculated in 0.5321 ms)

calc (CuPy(CUDA)/CuPyX.SciPy)> a = xp.arange(1, 10)
Result:
[1 2 3 4 5 6 7 8 9]
(Calculated in 0.2100 ms)

calc (CuPy(CUDA)/CuPyX.SciPy)> xp.mean(a)
Result:
5.0
(Calculated in 0.1543 ms)

calc (CuPy(CUDA)/CuPyX.SciPy)> sin(pi/2)  # Some common functions might be directly available
Result:
1.0
(Calculated in 0.1329 ms)

calc (CuPy(CUDA)/CuPyX.SciPy)> m = xp.array([[1,2],[3,4]])
Result:
[[1 2]
 [3 4]]
(Calculated in 0.0875 ms)

calc (CuPy(CUDA)/CuPyX.SciPy)> linalg.det(m)
Result:
-2.0000000000000004
(Calculated in 0.6543 ms)

calc (CuPy(CUDA)/CuPyX.SciPy)> fft.fft(xp.array([1,0,1,0]))
Result:
[2.+0.j 0.+0.j 2.+0.j 0.+0.j]
(Calculated in 0.2345 ms)

calc (CuPy(CUDA)/CuPyX.SciPy)> special.gamma(5) # Factorial of 4
Result:
24.0
(Calculated in 0.1120 ms)

calc (CuPy(CUDA)/CuPyX.SciPy)> random.rand(2,2) # Creates a 2x2 array of random numbers
Result:
[[0.123... 0.456...]
 [0.789... 0.321...]]
(Calculated in 0.0999 ms)

calc (CuPy(CUDA)/CuPyX.SciPy)> constants.c # Speed of light from scipy.constants
Result:
299792458.0
(Calculated in 0.0150 ms)

calc (CuPy(CUDA)/CuPyX.SciPy)> mode cpu
Switched to CPU (NumPy/SciPy) mode.
calc (NumPy/SciPy)> xp.sum(a)
Result:
45
(Calculated in 0.0312 ms)
```

### Available Functions & Modules (Summary)

Type `help` in the calculator for a more detailed guide. Here's a brief overview:

*   **Core Syntax & Ops:** `+`, `-`, `*`, `/`, `//`, `%`, `**`, `()`.
*   **Complex Numbers:** `complex(real, imag)` or `1j`.
*   **Array Creation:** `array()`, `arange()`, `linspace()`, `zeros()`, `ones()`, `eye()`, `diag()`, etc. (usually via `xp.`).
*   **Constants:** `pi`, `e`, `inf`, `nan`. SciPy constants via `constants.` (e.g., `constants.G`).
*   **General Math (on scalars or arrays):** `abs()`, `sqrt()`, `exp()`, `log()`, `log10()`, `power()`, `round()`, `degrees()`, `radians()`, etc. (often via `xp.` or directly).
*   **Trigonometric:** `sin()`, `cos()`, `tan()`, `asin()`, `acos()`, `atan()`, `atan2()`, etc.
*   **Hyperbolic:** `sinh()`, `cosh()`, `tanh()`, etc.
*   **Array Operations (use `xp.`):** `sum()`, `mean()`, `std()`, `min()`, `max()`, `dot()`, `matmul()`, `transpose()` or `.T`, `real()`, `imag()`, `concatenate()`, `stack()`, `unique()`, `where()`, etc.
*   **Linear Algebra (use `linalg.`):** `inv()`, `solve()`, `det()`, `eig()`, `svd()`, `norm()`, `qr()`, etc.
*   **FFT (use `fft.`):** `fft()`, `ifft()`, `rfft()`, `fftfreq()`, etc.
*   **Special Functions (use `special.`):** `gamma()`, `erf()`, `beta()`, `zeta()`, `comb()`, `perm()`, etc.
*   **Random Numbers (use `random.`):** `rand()`, `randn()`, `randint()`, `choice()`, `shuffle()`, etc.
*   **Statistics (use `stats.`):** Access to probability distributions (`norm`, `t`, etc.) for PDF, CDF, PPF, random variates. Functions like `ttest_ind()`, `pearsonr()`. (Primarily SciPy-based).

## Troubleshooting

*   **`CRITICAL ERROR: NumPy is not installed.`**:
    NumPy is essential. Install it: `pip install numpy`

*   **`CuPy (for GPU acceleration) or its SciPy components not found/configured.`**:
    *   This means CuPy is not installed or could not be imported correctly.
    *   Ensure you have compatible GPU drivers (NVIDIA or AMD).
    *   Install the CUDA Toolkit (for NVIDIA) or ROCm SDK (for AMD).
    *   Install the correct CuPy package for your CUDA/ROCm version (see Installation section).
    *   You can try `import cupy; cupy.show_config()` in a Python interpreter to diagnose CuPy issues.
    *   The calculator will fall back to CPU mode.

*   **`SciPy not found. Some advanced functions ... will be unavailable on CPU.`**:
    SciPy provides many advanced functions. Install it for full CPU functionality: `pip install scipy`

*   **`Error: Name '...' is not defined.`**:
    *   Check your spelling.
    *   Many functions need a prefix:
        *   `xp.some_array_function()`
        *   `linalg.some_linear_algebra_function()`
        *   `special.some_special_function()`
        *   `fft.some_fft_function()`
        *   `random.some_random_function()`
        *   `stats.some_stats_function_or_distribution()`
        *   `constants.some_constant`
    *   Type `help` to see available functions and modules.

*   **Errors during CuPy operations (e.g., `cupy.cuda.runtime.CUDARuntimeError`)**:
    *   This often indicates an issue with your CUDA/ROCm installation, driver compatibility, or CuPy installation.
    *   Ensure your GPU drivers are up to date.
    *   Verify that the installed CuPy version matches your CUDA/ROCm version.
    *   Check if your GPU is recognized by CuPy (`cupy.cuda.runtime.getDeviceCount() > 0`).
    *   You might be running out of GPU memory for very large operations.

## Contributing

Feel free to fork this project, submit issues, or suggest improvements.

## License

This project is open-source and can be considered under a permissive license like MIT if one were to be formally applied.
