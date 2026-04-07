# Extension modules
#
# types is the only non-math extension (no libm needed).
# All others are math-heavy and link libm.

py.extension_module(
    'types',
    'types.pyx',
    dependencies: [numpy_dep],
    include_directories: [pydgq_inc],
    install: true,
    subdir: 'pydgq/solver',
)

py.extension_module(
    'compsum',
    'compsum.pyx',
    dependencies: [numpy_dep, m_dep],
    include_directories: [pydgq_inc],
    install: true,
    subdir: 'pydgq/solver',
)

py.extension_module(
    'kernel_interface',
    'kernel_interface.pyx',
    dependencies: [numpy_dep, m_dep],
    include_directories: [pydgq_inc],
    install: true,
    subdir: 'pydgq/solver',
)

py.extension_module(
    'integrator_interface',
    'integrator_interface.pyx',
    dependencies: [numpy_dep, m_dep],
    include_directories: [pydgq_inc],
    install: true,
    subdir: 'pydgq/solver',
)

py.extension_module(
    'explicit',
    'explicit.pyx',
    dependencies: [numpy_dep, m_dep],
    include_directories: [pydgq_inc],
    install: true,
    subdir: 'pydgq/solver',
)

py.extension_module(
    'implicit',
    'implicit.pyx',
    dependencies: [numpy_dep, m_dep],
    include_directories: [pydgq_inc],
    install: true,
    subdir: 'pydgq/solver',
)

py.extension_module(
    'galerkin',
    'galerkin.pyx',
    dependencies: [numpy_dep, m_dep],
    include_directories: [pydgq_inc],
    install: true,
    subdir: 'pydgq/solver',
)

py.extension_module(
    'builtin_kernels',
    'builtin_kernels.pyx',
    dependencies: [numpy_dep, m_dep],
    include_directories: [pydgq_inc],
    install: true,
    subdir: 'pydgq/solver',
)

py.extension_module(
    'odesolve',
    'odesolve.pyx',
    dependencies: [numpy_dep, m_dep],
    include_directories: [pydgq_inc],
    install: true,
    subdir: 'pydgq/solver',
)

# Install Python files
py.install_sources(
    '__init__.py',
    subdir: 'pydgq/solver',
)

# CRITICAL: Install .pxd files for downstream cimport.
# Without these, user Cython code (and pydgq.examples) cannot cimport pydgq.solver.
# Also includes inline-only .pxd files (fputils, cminmax) that have no .pyx counterpart.
py.install_sources(
    'types.pxd',
    'compsum.pxd',
    'fputils.pxd',
    'cminmax.pxd',
    'kernel_interface.pxd',
    'integrator_interface.pxd',
    'explicit.pxd',
    'implicit.pxd',
    'galerkin.pxd',
    'builtin_kernels.pxd',
    subdir: 'pydgq/solver',
)
