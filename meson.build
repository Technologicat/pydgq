project(
    'pydgq',
    'cython', 'c',
    version: files('pydgq/VERSION'),
    license: 'BSD-2-Clause',
    meson_version: '>=1.3.0',
    default_options: [
        'buildtype=release',
        'c_std=c11',
        'cython_language=c',
    ],
)

py = import('python').find_installation(pure: false)

# NumPy include directory (needed for Cython compilation against NumPy)
numpy_dep = dependency('numpy')

# libm (needed for math-heavy extensions)
# On Linux this links -lm; on macOS/Windows it's a no-op (math is in libc/MSVCRT).
cc = meson.get_compiler('c')
m_dep = cc.find_library('m', required: false)

# Set Cython language level globally (avoid missing a file)
add_project_arguments('-X', 'language_level=3', language: 'cython')

# Windows: math.h requires _USE_MATH_DEFINES for M_PI (harmless no-op elsewhere)
add_project_arguments('-D_USE_MATH_DEFINES', language: 'c')

# Source root on include path for cimport resolution across subpackages
pydgq_inc = include_directories('.')

subdir('pydgq')
