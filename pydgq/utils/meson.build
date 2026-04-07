# Extension modules

py.extension_module(
    'discontify',
    'discontify.pyx',
    dependencies: [numpy_dep, m_dep],
    include_directories: [pydgq_inc],
    install: true,
    subdir: 'pydgq/utils',
)

# Install Python files
py.install_sources(
    '__init__.py',
    'precalc.py',
    'listutils.py',
    'mpi_shim.py',
    subdir: 'pydgq/utils',
)
