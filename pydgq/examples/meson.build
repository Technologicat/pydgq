py.extension_module(
    'example_kernel',
    'example_kernel.pyx',
    dependencies: [numpy_dep, m_dep],
    include_directories: [pydgq_inc],
    install: true,
    subdir: 'pydgq/examples',
)

py.install_sources(
    '__init__.py',
    subdir: 'pydgq/examples',
)

# Install .pxd for users who want to subclass MyKernel
py.install_sources(
    'example_kernel.pxd',
    subdir: 'pydgq/examples',
)
