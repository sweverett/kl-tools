# Comparison of intensity.py: Current Branch vs tng_grism Branch

## Executive Summary

This document summarizes the key differences between `kl_tools/intensity.py` in the current branch (copilot/compare-intensity-files) and the tng_grism branch. The files show significant architectural differences, with the tng_grism branch representing an older, more complex implementation.

**Overall Statistics:**
- Current branch: 1,279 lines
- tng_grism branch: 1,275 lines
- Changes: ~750 insertions, ~754 deletions (substantial refactoring)

---

## Major Architectural Differences

### 1. **File Structure and Organization**

#### Current Branch (Modern)
- Clean, focused imports
- No command-line argument parsing
- Pure library code with clear abstractions
- Well-documented docstrings with type hints

#### tng_grism Branch (Legacy)
- Includes debugging imports (`ipdb`)
- Has ArgumentParser for command-line testing
- Includes a `main()` function for testing (lines 1004-1275)
- More comments about development status

---

## Class-by-Class Comparison

### 2. **IntensityMap Base Class**

#### Current Branch
```python
def __init__(self, name, continuum=None):
```
- **Modern approach**: Removed image size parameters (nx, ny) from constructor
- **Continuum handling**: Sophisticated handling with support for:
  - Static numpy arrays
  - Callable functions (dynamic computation)
  - None (defaults to zeros)
- **New features**:
  - `set_continuum()` method for updating continuum
  - `_continuum_callable` flag for dynamic continuum
- **render() signature**:
  ```python
  render(self, image_pars, theta_pars, pars, weights=None, mask=None, 
         image=None, redo=True, im_type='emission', raise_errors=True)
  ```
  - Uses `image_pars` object (ImagePars) instead of direct datacube
  - Added `weights` and `mask` support
  - Added `raise_errors` parameter for error handling
  - More flexible with `image` parameter

#### tng_grism Branch
```python
def __init__(self, name, nx, ny):
```
- **Legacy approach**: Requires explicit image dimensions in constructor
- **Simpler continuum**: Just a basic attribute
- **render() signature**:
  ```python
  render(self, theta_pars, datacube, pars, redo=False, im_type='emission')
  ```
  - Directly uses `datacube` object
  - No weights/mask support
  - Simpler error handling
  - Special handling for `imap_return_gal` option

---

### 3. **InclinedExponential Class**

#### Current Branch
```python
def __init__(self, flux, hlr=None, scale_radius=None, continuum=None):
```
- **Clean parameters**: Flux and size (hlr or scale_radius)
- **Flexible sizing**: Can use either `hlr` (half-light radius) or `scale_radius`
- **_render() parameters**: 
  - Uses `image_pars` object
  - Returns tuple: `(image, continuum)`
  - PSF handling through `pars` dictionary
  - Uses `theta_pars['x0']` and `theta_pars['y0']` for offsets
  - Explicit handling of PSF convolution in render

#### tng_grism Branch
```python
def __init__(self, datavector, kwargs):
```
- **Complex initialization**: 
  - Takes entire `datavector` (datacube) object
  - Uses kwargs dictionary for all parameters
  - Extracts dimensions from datacube or kwargs
- **Multiple emission lines**: 
  - Supports per-line hlr (e.g., `em_PaA_hlr`)
  - Supports continuum per line (e.g., `cont_{emline}_hlr`)
  - Renders separate images for each emission line
- **Advanced features**:
  - Disk and spectroscopic offsets: `dx_disk`, `dy_disk`, `dx_spec`, `dy_spec`
  - Returns dictionary of images (not single image): `self.image = {}`
  - Separate Sersic profiles for continuum (n=4)
  - Timing instrumentation (`from time import time`)
  - Error handling with `gs.GalSimFFTSizeError`

**Key Insight**: tng_grism version is designed for multi-line spectroscopy with separate profiles per line.

---

### 4. **BasisIntensityMap Class**

#### Current Branch
```python
def __init__(self, basis_type, basis_kwargs=None, basis_plane='obs', 
             continuum=None, skip_ground_state=False):
```
- **Modern design**:
  - No datacube dependency in constructor
  - Explicit `basis_plane` parameter
  - `skip_ground_state` feature for residual fitting
  - Continuum is a template parameter
- **Key features**:
  - `get_basis()` method and `basis` property
  - `plane` property
  - Lazy fitter initialization (setup during render)
  - Returns tuple: `(image, continuum)`

#### tng_grism Branch
```python
def __init__(self, datacube, basis_type='default', basis_kwargs=None):
```
- **Legacy design**:
  - Requires datacube in constructor
  - Extracts dimensions and pixel scale from datacube
  - Immediate fitter setup during init
  - Continuum template built from datacube method: `datacube.get_continuum()`
  - PSF from datacube: `datacube.get_psf()`
- **Features**:
  - Adaptive moment finding (commented out): `gs.hsm.FindAdaptiveMom`
  - `remove_continuum` option in fit method
  - More tightly coupled to datacube structure

---

### 5. **CompositeIntensityMap Class**

#### Current Branch
- **NEW CLASS**: Not present in tng_grism branch
- **Purpose**: Combines InclinedExponential + BasisIntensityMap
- **Use case**: PSF convolution of composite models
- **Key method**: `_render()` combines exponential and basis components

#### tng_grism Branch
- Does not exist

---

### 6. **GMixModel Class**

#### Current Branch
- Does not exist

#### tng_grism Branch
- **NEW CLASS**: Mixture of Gaussians model
- **Purpose**: Fit inclined exponential or Sersic using Gaussian mixture (GMix)
- **Features**:
  - Two components: emission line (exponential disk) and continuum (de Vaucouleurs)
  - Based on NGMIX (Sheldon 2014) and Hogg & Lang (2012)
  - Projection handling: converts inclination to ellipticity
  - Shear transform with analytical solution (Bernstein & Jarvis 2002)
- **Status**: Implementation incomplete (just `pass` in methods)

---

### 7. **IntensityMapFitter Class**

#### Current Branch
```python
def __init__(self, basis_type, basis_kwargs, basis_plane, image_pars, 
             continuum_template=None, skip_ground_state=False, psf=None, 
             weights=None, mask=None):
```
- **ImagePars-based**: Uses `image_pars` object
- **Comprehensive features**:
  - Weights and mask support
  - Skip ground state option
  - Explicit basis plane handling
  - Grid built with `indexing='xy'`
- **Rendering**:
  - `basis.render_im()` signature: `render_im(coeffs, image_pars, plane, transformation_pars)`
  - Returns proper shape checking
  - Handles `skip_ground_state` by inserting zero coefficient

#### tng_grism Branch
```python
def __init__(self, basis_type, nx, ny, continuum_template=None, 
             psf=None, basis_kwargs=None):
```
- **Dimension-based**: Takes nx, ny directly
- **Simpler design**:
  - No weights/mask
  - No skip_ground_state
  - Basis plane from basis object
  - Grid built with default indexing
- **Rendering**:
  - `basis.render_im()` signature: `render_im(theta_pars, coeffs)`
  - Data flattened with `reshape()` not `flatten()`
  - Simpler coefficient handling

---

### 8. **Helper Functions**

#### `build_intensity_map()`

**Current Branch**:
```python
def build_intensity_map(name, kwargs):
```
- Takes just name and kwargs
- No datavector dependency

**tng_grism Branch**:
```python
def build_intensity_map(name, datavector, kwargs):
```
- Requires datavector parameter
- All constructors expect datavector

#### `fit_for_beta()`

**Current Branch** (lines 1202-1278):
- More complete implementation
- Supports parallel processing: `ncores` parameter
- Uses `Pool` from multiprocessing
- Calls `fit_one_beta()` helper

**tng_grism Branch** (lines 977-1002):
- Incomplete stub implementation
- Simple TODO comment
- No parallelization

---

## Key Conceptual Differences

### 1. **Dependency Injection vs. Direct Coupling**
- **Current**: Classes don't require datacube; use `image_pars` abstraction
- **tng_grism**: Classes tightly coupled to datacube structure

### 2. **Error Handling**
- **Current**: `raise_errors` parameter for graceful degradation
- **tng_grism**: Uses try/except for `GalSimFFTSizeError` with zero fallback

### 3. **Continuum Handling**
- **Current**: Sophisticated (static/callable/None), separated from emission
- **tng_grism**: Simple template from datacube, per-line support

### 4. **Multi-line Support**
- **Current**: Single emission line focus, cleaner abstraction
- **tng_grism**: Built-in multi-line support with per-line parameters

### 5. **PSF Convolution**
- **Current**: Handled in `_render()`, can apply to composite models
- **tng_grism**: Simpler, per-component approach

### 6. **Code Maturity**
- **Current**: Production-ready, no debug code, no test harness
- **tng_grism**: Development code with `ipdb`, testing main(), argparser

---

## Import Differences

### Current Branch
```python
import kl_tools.utils as utils
import kl_tools.basis as basis
import kl_tools.likelihood as likelihood
from kl_tools.transformation import transform_coords, SUPPORTED_PLANES
```

### tng_grism Branch
```python
import os
from time import time
import kl_tools.parameters as parameters
from kl_tools.emission import LINE_LAMBDAS
from kl_tools.transformation import transform_coords, TransformableImage
import ipdb
```

**Key differences**:
- tng_grism includes development imports: `ipdb`, `os`, `time`
- tng_grism imports `parameters` module and `LINE_LAMBDAS`
- Current imports `SUPPORTED_PLANES` constant
- tng_grism imports `TransformableImage` (unused in shown code)

---

## INTENSITY_TYPES Registry

### Current Branch
```python
INTENSITY_TYPES = {
    'default': BasisIntensityMap,
    'basis': BasisIntensityMap,
    'inclined_exp': InclinedExponential,
}
```

### tng_grism Branch
```python
INTENSITY_TYPES = {
    'default': BasisIntensityMap,
    'basis': BasisIntensityMap,
    'inclined_exp': InclinedExponential,
    'mogs': GMixModel,
}
```

**Difference**: tng_grism includes `GMixModel` (though incomplete)

---

## Code Quality and Style

### Current Branch
- ✅ Clean, production-ready code
- ✅ Comprehensive docstrings
- ✅ Type hints in docstrings
- ✅ Consistent error messages
- ✅ No debugging code
- ✅ Modular design

### tng_grism Branch
- ⚠️ Contains debugging artifacts (`ipdb`, timing)
- ⚠️ Includes test harness in main file
- ⚠️ Some incomplete features (GMixModel)
- ⚠️ Commented-out code blocks
- ✅ Good documentation
- ⚠️ More complex due to multi-line support

---

## Migration Path

If moving from tng_grism to current branch, key changes needed:

1. **Constructor updates**:
   - Remove `datavector` parameter
   - Use `ImagePars` instead of extracting from datacube
   - Update kwargs structure

2. **Render method updates**:
   - Change `datacube` to `image_pars`
   - Add `image` parameter
   - Update return handling (tuples vs. optional gal return)

3. **Multi-line support**:
   - Need to handle per-line parameters differently
   - May need multiple IntensityMap instances

4. **Error handling**:
   - Use `raise_errors` parameter
   - Remove GalSimFFTSizeError zero-fallback pattern

5. **Continuum**:
   - Set via `continuum` parameter or `set_continuum()`
   - No longer from datacube.get_continuum()

---

## Recommendations

### Use Current Branch If:
- Building new features
- Need cleaner abstraction from datacube
- Single emission line is primary use case
- Want production-ready code
- Need composite models (exp + basis)
- Want better testing isolation

### Use tng_grism Branch If:
- Working with multi-line spectroscopy
- Need per-line morphology (different hlr per line)
- Working with existing tng_grism pipeline
- Need GMix model foundation (even if incomplete)
- Have legacy code depending on datacube coupling

---

## Summary

The current branch represents a **significant refactoring** toward:
1. Better separation of concerns (ImagePars vs datacube)
2. More flexible continuum handling
3. Cleaner, production-ready code
4. Support for composite models
5. Better testability

The tng_grism branch has:
1. Built-in multi-line spectroscopy support
2. Tighter integration with datacube structure
3. More development/experimental code
4. Foundation for GMix models

**Overall**: Current branch is a **modernization** and **simplification** suitable for single-line analysis, while tng_grism is more **feature-rich for multi-line spectroscopy** but with more complexity and coupling.
