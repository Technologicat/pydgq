# Install Python files and version
py.install_sources(
    '__init__.py',
    'data.py',
    'VERSION',
    subdir: 'pydgq',
)

# Install precalculated data file
py.install_sources(
    'pydgq_data.npz',
    subdir: 'pydgq',
)

subdir('solver')
subdir('utils')
subdir('examples')
