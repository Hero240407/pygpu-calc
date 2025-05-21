import sys
import math
import cmath # For scalar complex math, though numpy/cupy handle complex arrays
import time

# --- Attempt to import NumPy (essential) ---
try:
    import numpy as np
except ImportError:
    print("CRITICAL ERROR: NumPy is not installed. This calculator cannot run.")
    print("Please install it using: pip install numpy")
    sys.exit(1)

# --- Attempt to import CuPy (for GPU) ---
try:
    import cupy as cp
    import cupyx.scipy.special as cp_special
    import cupyx.scipy.linalg as cp_linalg
    # cupyx.scipy.fft is often used instead of cupy.fft for more SciPy-like API
    import cupyx.scipy.fft as cp_fft
    CUPY_AVAILABLE = True
    # Check if CuPy is using CUDA or ROCm (HIP)
    # This is a heuristic; cupy.show_config() is more definitive but verbose for startup
    backend_name = "CUDA"
    try:
        if cp.cuda.runtime.getDeviceCount() > 0: # Checks for CUDA devices
             pass # Default backend_name is CUDA
    except cp.cuda.runtime.CUDARuntimeError: # If CUDA runtime error, it might be ROCm or no GPU
        try:
            # Check for HIP devices if CUDA check failed or to be more specific
            # cupy.hip.runtime.getDeviceCount() would be ideal if universally available in cp namespace
            # For now, we rely on the fact that if cupy imported, it has *some* backend.
            # A more robust check might involve looking at cupy.show_config() output or specific hip attributes
            # This simplified check assumes if not CUDA and CuPy imported, it could be ROCm
            # but without a direct cp.hip.runtime.getDeviceCount(), it's hard to be certain here.
            # We will assume CuPy handles its backend.
            if hasattr(cp, 'hip') and cp.hip.runtime.getDeviceCount() > 0:
                backend_name = "ROCm/HIP"
        except Exception:
            pass # Stick to CUDA as default if HIP check fails

    print(f"CuPy and CuPyX.SciPy components found. GPU acceleration (likely via {backend_name}) enabled where applicable.")

except ImportError:
    CUPY_AVAILABLE = False
    cp = None # Ensure cp is defined for type hinting or checks
    cp_special = None
    cp_linalg = None
    cp_fft = None
    print("CuPy (for GPU acceleration) or its SciPy components not found/configured.")
    print("  - If you have an NVIDIA GPU, ensure CUDA and CuPy are correctly installed.")
    print("  - If you have an AMD GPU, CuPy needs to be installed with ROCm support.")
    print("Falling back to NumPy/SciPy (CPU) for all operations.")
except Exception as e: # Catch other potential errors during CuPy import/check
    CUPY_AVAILABLE = False
    cp = None
    cp_special = None
    cp_linalg = None
    cp_fft = None
    print(f"An error occurred during CuPy initialization: {e}")
    print("Falling back to NumPy/SciPy (CPU) for all operations.")


# --- Attempt to import SciPy (for advanced CPU functions) ---
try:
    import scipy
    import scipy.special
    import scipy.linalg
    import scipy.fft
    import scipy.stats
    import scipy.constants
    SCIPY_AVAILABLE = True
    print("SciPy found. Advanced CPU functions available.")
except ImportError:
    SCIPY_AVAILABLE = False
    # Define dummy modules/attributes if scipy is not available to prevent NameErrors later
    class DummyModule: pass
    scipy = DummyModule() # type: ignore
    scipy.special = None # type: ignore
    scipy.linalg = None # type: ignore
    scipy.fft = None # type: ignore
    scipy.stats = None # type: ignore
    scipy.constants = None # type: ignore
    print("SciPy not found. Some advanced functions (special, advanced linalg/fft, stats, constants) will be unavailable on CPU.")


# --- Helper to check if we should use GPU ---
def use_gpu_if_available(prefer_gpu=True):
    return CUPY_AVAILABLE and prefer_gpu

# --- Define the safe execution environment ---
SAFE_EVAL_NAMESPACE = {}

def populate_namespace(use_gpu):
    global SAFE_EVAL_NAMESPACE
    SAFE_EVAL_NAMESPACE = {} # Reset

    current_xp = None
    if use_gpu and CUPY_AVAILABLE:
        current_xp = cp
        SAFE_EVAL_NAMESPACE['xp'] = cp
        SAFE_EVAL_NAMESPACE['cp'] = cp # Explicitly allow 'cp'
        # print("Using CuPy for base array operations.")
    else:
        current_xp = np
        SAFE_EVAL_NAMESPACE['xp'] = np
        SAFE_EVAL_NAMESPACE['np'] = np # Explicitly allow 'np'
        # print("Using NumPy for base array operations.")

    # Python's math and cmath modules
    SAFE_EVAL_NAMESPACE['math'] = math
    SAFE_EVAL_NAMESPACE['cmath'] = cmath

    # Core constants
    SAFE_EVAL_NAMESPACE['pi'] = current_xp.pi if hasattr(current_xp, 'pi') else math.pi
    SAFE_EVAL_NAMESPACE['e'] = current_xp.e if hasattr(current_xp, 'e') else math.e
    SAFE_EVAL_NAMESPACE['inf'] = current_xp.inf if hasattr(current_xp, 'inf') else float('inf')
    SAFE_EVAL_NAMESPACE['nan'] = current_xp.nan if hasattr(current_xp, 'nan') else float('nan')

    # Basic Python built-ins
    SAFE_EVAL_NAMESPACE['True'] = True
    SAFE_EVAL_NAMESPACE['False'] = False
    SAFE_EVAL_NAMESPACE['None'] = None
    safe_builtins = ['len', 'round', 'complex', 'float', 'int', 'str', 'list', 'tuple', 'dict', 'set', 'abs', 'sum', 'min', 'max']
    for b_name in safe_builtins:
        if hasattr(__builtins__, b_name): # Ensure it's a real builtin
             SAFE_EVAL_NAMESPACE[b_name] = getattr(__builtins__, b_name)


    # --- NumPy/CuPy core functions ---
    # (many will be aliased from current_xp for convenience)
    numpy_cupy_functions = [
        'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
        'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
        'exp', 'exp2', 'expm1', 'log', 'log2', 'log10', 'log1p',
        'sqrt', 'cbrt', 'square', 'reciprocal',
        'power', 'float_power',
        'abs', 'absolute', 'fabs', 'sign', # 'abs' is also a builtin
        'ceil', 'floor', 'trunc', 'rint',
        'degrees', 'radians',
        'sum', 'prod', 'mean', 'median', 'std', 'var', # 'sum' is also a builtin
        'min', 'max', 'amin', 'amax', 'ptp', # 'min', 'max' are also builtins
        'argmin', 'argmax', 'nanargmin', 'nanargmax', 'nanmin', 'nanmax', 'nansum', 'nanmean', 'nanstd', 'nanvar',
        'array', 'asarray', 'arange', 'linspace', 'logspace',
        'zeros', 'ones', 'empty', 'full', 'eye', 'identity', 'diag', 'diagflat',
        'dot', 'vdot', 'inner', 'outer', 'matmul', 'linalg', # linalg submodule handled below
        'transpose', 'roll', 'reshape', 'ravel', 'flatten',
        'concatenate', 'stack', 'hstack', 'vstack', 'dstack',
        'split', 'array_split', 'hsplit', 'vsplit', 'dsplit',
        'kron', 'tile', 'repeat',
        'sort', 'argsort', 'searchsorted', 'unique', 'in1d', 'intersect1d', 'union1d', 'setdiff1d', 'setxor1d',
        'where', 'select', 'extract', 'count_nonzero',
        'diff', 'gradient', 'cumsum', 'cumprod', 'convolve', 'correlate',
        'real', 'imag', 'conj', 'angle',
        'unwrap', 'isreal', 'iscomplex', 'isclose', 'allclose',
        'fft' # fft submodule handled below
    ]
    for func_name in numpy_cupy_functions:
        if hasattr(current_xp, func_name):
            SAFE_EVAL_NAMESPACE[func_name] = getattr(current_xp, func_name)

    # Random submodule
    if hasattr(current_xp, 'random'):
        SAFE_EVAL_NAMESPACE['random'] = getattr(current_xp, 'random')

    # --- SciPy / CuPyX.SciPy modules ---
    if use_gpu and CUPY_AVAILABLE:
        if cp_linalg: SAFE_EVAL_NAMESPACE['linalg'] = cp_linalg
        elif hasattr(cp, 'linalg'): SAFE_EVAL_NAMESPACE['linalg'] = cp.linalg # Fallback to cupy.linalg
        else: print("Warning: cp.linalg and cupyx.scipy.linalg not found for GPU.")

        if cp_fft: SAFE_EVAL_NAMESPACE['fft'] = cp_fft
        elif hasattr(cp, 'fft'): SAFE_EVAL_NAMESPACE['fft'] = cp.fft
        else: print("Warning: cp.fft and cupyx.scipy.fft not found for GPU.")

        if cp_special: SAFE_EVAL_NAMESPACE['special'] = cp_special
        else: print("Warning: cupyx.scipy.special not found for GPU.")
        
        # For stats and constants, CuPyX doesn't have full direct equivalents for all of SciPy's modules.
        # We can provide SciPy's CPU versions if available, for broader functionality.
        if SCIPY_AVAILABLE and scipy.stats: SAFE_EVAL_NAMESPACE['stats'] = scipy.stats
        else: print("Warning: scipy.stats not found (CPU fallback for stats).")
        if SCIPY_AVAILABLE and scipy.constants: SAFE_EVAL_NAMESPACE['constants'] = scipy.constants
        else: print("Warning: scipy.constants not found (CPU fallback for constants).")

    else: # CPU mode or SciPy fallback
        if SCIPY_AVAILABLE:
            if scipy.linalg: SAFE_EVAL_NAMESPACE['linalg'] = scipy.linalg
            elif hasattr(np, 'linalg'): SAFE_EVAL_NAMESPACE['linalg'] = np.linalg # Fallback to numpy.linalg
            else: print("Warning: np.linalg and scipy.linalg not found for CPU.")

            if scipy.fft: SAFE_EVAL_NAMESPACE['fft'] = scipy.fft
            elif hasattr(np, 'fft'): SAFE_EVAL_NAMESPACE['fft'] = np.fft
            else: print("Warning: np.fft and scipy.fft not found for CPU.")

            if scipy.special: SAFE_EVAL_NAMESPACE['special'] = scipy.special
            else: print("Warning: scipy.special not found for CPU.")

            if scipy.stats: SAFE_EVAL_NAMESPACE['stats'] = scipy.stats
            else: print("Warning: scipy.stats not found for CPU.")

            if scipy.constants: SAFE_EVAL_NAMESPACE['constants'] = scipy.constants
            else: print("Warning: scipy.constants not found for CPU.")
        else: # SciPy not available, try to provide basic linalg/fft from numpy
            if hasattr(np, 'linalg'): SAFE_EVAL_NAMESPACE['linalg'] = np.linalg
            if hasattr(np, 'fft'): SAFE_EVAL_NAMESPACE['fft'] = np.fft
            print("SciPy not available, linalg/fft will be basic NumPy versions. Special functions, stats, constants limited on CPU.")

    # Convenience: some popular functions directly in namespace
    # Overwrites builtins like sum, min, max with xp versions for array handling
    convenience_functions = [
        'sin', 'cos', 'tan', 'sqrt', 'log', 'log10', 'exp', 'array', 'arange', 'linspace',
        'sum', 'mean', 'std', 'min', 'max', 'dot', 'abs', 'round', 'power', 'degrees', 'radians'
    ]
    for func_name in convenience_functions:
        if hasattr(current_xp, func_name):
             SAFE_EVAL_NAMESPACE[func_name] = getattr(current_xp, func_name)
        elif func_name in SAFE_EVAL_NAMESPACE and hasattr(SAFE_EVAL_NAMESPACE[func_name], func_name): # e.g. math.sin
             pass # already there or from math
        elif hasattr(math, func_name) and not hasattr(current_xp, func_name): # for scalar fallbacks if not in xp
             SAFE_EVAL_NAMESPACE[func_name] = getattr(math, func_name)


def calculate(expression_str, use_gpu_preference):
    populate_namespace(use_gpu_if_available(use_gpu_preference))
    result = None
    duration_ms = -1.0

    try:
        active_xp = cp if use_gpu_if_available(use_gpu_preference) and CUPY_AVAILABLE else np
        
        # Synchronize stream for accurate timing if using GPU.
        # CuPy maps cupy.cuda.* to cupy.hip.* if using ROCm backend.
        if use_gpu_if_available(use_gpu_preference) and CUPY_AVAILABLE:
            if hasattr(cp, 'cuda') and hasattr(cp.cuda, 'Stream') and hasattr(cp.cuda.Stream, 'null'):
                cp.cuda.Stream.null.synchronize()
            elif hasattr(cp, 'hip') and hasattr(cp.hip, 'Stream') and hasattr(cp.hip.Stream, 'null'): # Explicit HIP check just in case
                cp.hip.Stream.null.synchronize()

        start_time = time.perf_counter()

        # The __builtins__ override is critical for security with eval
        # Provide only the 'safe' builtins we populated earlier
        eval_builtins = {k: SAFE_EVAL_NAMESPACE[k] for k in 
                         ['len', 'round', 'complex', 'float', 'int', 'str', 'list', 'tuple', 'dict', 'set', 'abs', 'sum', 'min', 'max']
                         if k in SAFE_EVAL_NAMESPACE}
        eval_builtins['__import__'] = None # Explicitly disable __import__

        result = eval(expression_str, {"__builtins__": eval_builtins}, SAFE_EVAL_NAMESPACE)

        # Synchronize stream again after computation if using GPU.
        if use_gpu_if_available(use_gpu_preference) and CUPY_AVAILABLE and isinstance(result, cp.ndarray):
            if hasattr(cp, 'cuda') and hasattr(cp.cuda, 'Stream') and hasattr(cp.cuda.Stream, 'null'):
                cp.cuda.Stream.null.synchronize()
            elif hasattr(cp, 'hip') and hasattr(cp.hip, 'Stream') and hasattr(cp.hip.Stream, 'null'):
                cp.hip.Stream.null.synchronize()

        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        return result, duration_ms
    except NameError as e:
        # Try to provide a more helpful message by listing some available top-level names
        available_keys = [k for k in SAFE_EVAL_NAMESPACE.keys() if not k.startswith('_')]
        # Filter out module objects themselves if their contents are more relevant or too verbose
        filtered_keys = [k for k in available_keys if not isinstance(SAFE_EVAL_NAMESPACE[k], type(math))]
        # Add submodule names explicitly
        for sm in ['linalg', 'fft', 'special', 'stats', 'random', 'constants']:
            if sm in SAFE_EVAL_NAMESPACE:
                filtered_keys.append(sm)
        
        # Get a sample of functions from xp for hint
        xp_funcs_sample = []
        if 'xp' in SAFE_EVAL_NAMESPACE:
            xp_module = SAFE_EVAL_NAMESPACE['xp']
            try:
                xp_funcs_sample = [f"xp.{f}" for f in dir(xp_module) if not f.startswith('_') and callable(getattr(xp_module, f, None))][:5]
            except Exception: # Handle cases where dir(xp_module) might be problematic (e.g. if xp is None)
                pass

        return f"Error: Name '{e.name}' is not defined. Check spelling or if it needs a prefix (e.g., xp.{e.name}, linalg.{e.name}).\nAvailable top-level (sample): {', '.join(sorted(list(set(filtered_keys))[:20]))}...\nExample xp functions: {', '.join(xp_funcs_sample)}...", duration_ms
    except SyntaxError as e:
        return f"Error: Syntax error in expression. {e}", duration_ms
    except TypeError as e:
        return f"Error: Type error in expression. {e}", duration_ms
    except ZeroDivisionError as e:
        return f"Error: Division by zero. {e}", duration_ms
    except Exception as e:
        return f"An unexpected error occurred: {type(e).__name__} - {e}", duration_ms

def print_guide():
    print("\n--- Advanced GPU-Accelerated Python Calculator (Full Version) ---")
    print("Enter mathematical expressions using Python syntax and available functions.")
    print("Modes: 'mode gpu' (uses CuPy if available), 'mode cpu' (uses NumPy/SciPy).")
    print("  - For GPU mode with NVIDIA GPUs, CuPy with CUDA is used.")
    print("  - For GPU mode with AMD GPUs, CuPy must be installed with ROCm support.")
    print("Type 'exit' or 'quit' to close. Type 'help' for this guide again.")
    print("---------------------------------------------------------------------")
    print("QUICK GUIDE TO AVAILABLE FUNCTIONS & MODULES:")
    print("---------------------------------------------------------------------")
    print("PREFIXING: For many array functions, use 'xp.' (e.g., xp.mean(array)).")
    print("           'xp' automatically points to 'cp' (CuPy for GPU) or 'np' (NumPy for CPU) based on current mode.")
    print("           Some very common functions (sin, cos, array, log, etc.) might be usable directly.")
    print("           Access submodules like 'linalg.det()' or 'special.gamma()'.")
    print("\nCATEGORIES (Examples):")
    print("  Core Syntax & Ops: +, -, *, /, // (floor div), % (mod), ** (power), ()")
    print("  Complex Numbers: complex(real, imag) or 1j. E.g., sqrt(complex(-1,0))")
    print("  Array Creation: array([...]), arange(start, stop, step), linspace(start, stop, num),")
    print("                  logspace(...), zeros((rows,cols)), ones(...), eye(N), diag([...])")
    print("  Constants: pi, e, inf, nan. With SciPy: constants.c (speed of light), constants.G (gravity), etc.")
    print("\n  General Math (often on arrays via xp. or directly):")
    print("    abs(x), sqrt(x), cbrt(x), exp(x), log(x) (natural), log10(x), log2(x),")
    print("    power(base, exp), degrees(rad), radians(deg), round(x, decimals), floor(x), ceil(x)")
    print("  Trigonometric (input in radians):")
    print("    sin(x), cos(x), tan(x), asin(x), acos(x), atan(x), atan2(y,x)")
    print("  Hyperbolic Functions:")
    print("    sinh(x), cosh(x), tanh(x), asinh(x), acosh(x), atanh(x)")
    print("\n  Array Operations (use xp. prefix for most):")
    print("    sum, mean, median, std, var, min, max, ptp (peak-to-peak), diff, sort,")
    print("    cumsum, cumprod, dot, matmul, transpose(arr) or arr.T, real(arr), imag(arr), conj(arr)")
    print("    concatenate((a,b)), stack((a,b)), hstack, vstack, unique, where(condition, x, y)")
    print("\n  Linear Algebra (use 'linalg.' prefix, e.g., linalg.inv(arr)):")
    print("    inv, solve, det, eig, eigh, svd, norm, qr, cholesky, matrix_rank, lstsq")
    print("    (Availability depends on NumPy/SciPy/CuPyX.SciPy)")
    print("\n  FFT (Fast Fourier Transform) (use 'fft.' prefix, e.g., fft.fft(arr)):")
    print("    fft, ifft, fftn, ifftn, rfft, irfft, fftfreq, rfftfreq")
    print("    (Availability depends on NumPy/SciPy/CuPyX.SciPy)")
    print("\n  Special Functions (use 'special.' prefix, e.g., special.gamma(x)):")
    print("    gamma, gammaln, beta, erf, erfc, psi (digamma), zeta")
    print("    factorial (use math.factorial(scalar_int) or special.gamma(array_float+1))")
    print("    perm(N,k), comb(N,k) (permutations/combinations)")
    print("    (Availability depends on SciPy/CuPyX.SciPy)")
    print("\n  Random Numbers (use 'random.' prefix, e.g., random.rand(3,3)):")
    print("    rand (uniform), randn (normal), randint(low, high, size), choice(arr, size), shuffle(arr), permutation(arr)")
    print("\n  Statistics (use 'stats.' prefix, e.g., stats.norm.pdf(x, loc, scale)):")
    print("    Access to distributions (norm, t, chi2, etc.) for pdf, cdf, ppf, rvs.")
    print("    Basic stats: describe, ttest_1samp, ttest_ind, pearsonr, spearmanr")
    print("    (Primarily from SciPy, CPU-only for complex stats objects if CuPyX lacks them)")
    print("---------------------------------------------------------------------")


def main():
    print_guide()

    current_mode_gpu = True if CUPY_AVAILABLE else False # Default to GPU if available

    if not CUPY_AVAILABLE:
        print("\nNOTE: CuPy not found or not configured for your GPU (see messages above).")
        print("Running in CPU (NumPy/SciPy) mode only.")
    else:
        gpu_backend_info = "CuPy/CuPyX.SciPy (CUDA or ROCm/HIP if configured)"
        print(f"\nInitial mode: {'GPU (' + gpu_backend_info + ')' if current_mode_gpu else 'CPU (NumPy/SciPy)'}")
    
    # Initial population to ensure mode is set for the first prompt
    populate_namespace(use_gpu_if_available(current_mode_gpu))


    while True:
        try:
            # Display current processing library before prompt
            active_lib_base = "CuPy" if use_gpu_if_available(current_mode_gpu) and CUPY_AVAILABLE else "NumPy"
            active_lib_adv = ("CuPyX.SciPy" if use_gpu_if_available(current_mode_gpu) and CUPY_AVAILABLE and 
                              (cp_special or cp_linalg or cp_fft) else "SciPy")
            
            backend_hint = ""
            if active_lib_base == "CuPy":
                # Attempt to get a more specific backend hint for the prompt if CuPy is active
                try:
                    if hasattr(cp, 'cuda') and cp.cuda.runtime.getDeviceCount() > 0 :
                        backend_hint = "(CUDA)"
                    elif hasattr(cp, 'hip') and cp.hip.runtime.getDeviceCount() > 0:
                        backend_hint = "(ROCm/HIP)"
                except Exception: # Fallback if device check fails
                    backend_hint = "(GPU)"


            prompt_prefix = f"calc ({active_lib_base}{backend_hint}/{active_lib_adv})> "
            expression = input(prompt_prefix)

            if expression.lower() in ['exit', 'quit']:
                print("Exiting calculator.")
                break
            elif expression.lower() == 'help':
                print_guide()
                populate_namespace(use_gpu_if_available(current_mode_gpu))
                continue
            elif expression.lower() == 'mode gpu':
                if CUPY_AVAILABLE:
                    current_mode_gpu = True
                    populate_namespace(True) 
                    print("Switched to GPU (CuPy/CuPyX.SciPy if available) mode.")
                    print("  Ensure your CuPy installation matches your GPU (CUDA for NVIDIA, ROCm for AMD).")
                else:
                    print("CuPy not available. Cannot switch to GPU mode. Staying in CPU mode.")
                continue
            elif expression.lower() == 'mode cpu':
                current_mode_gpu = False
                populate_namespace(False)
                print("Switched to CPU (NumPy/SciPy) mode.")
                continue
            elif expression.strip() == "":
                continue

            result, duration_ms = calculate(expression, current_mode_gpu)
            
            print(f"Result:")
            if CUPY_AVAILABLE and isinstance(result, cp.ndarray):
                if result.size > 20 and result.ndim > 0 : 
                    print(f"CuPy Array (showing first/last few elements if large):\n{cp.array_str(result, max_line_width=120)}")
                else:
                    try:
                        print(cp.asnumpy(result))
                    except Exception as e:
                        print(f"(Could not convert CuPy array to NumPy for printing: {e})")
                        print(result) # Print raw CuPy array as fallback
            else:
                print(result)

            if duration_ms >= 0:
                print(f"(Calculated in {duration_ms:.4f} ms)")

        except EOFError:
            print("\nExiting...")
            break
        except KeyboardInterrupt:
            print("\nCalculation interrupted. Type 'exit' or 'quit'.")
            populate_namespace(use_gpu_if_available(current_mode_gpu)) # Re-populate
            pass

if __name__ == "__main__":
    if 'np' not in globals() or np is None: # Check if NumPy actually loaded
         print("CRITICAL: NumPy failed to load properly. Please check your installation.")
    else:
        main()