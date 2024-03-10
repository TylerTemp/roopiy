import sys
from setuptools import setup
from roopiy.version import __version__

setup(
    name='roopiy',
    version=__version__,
    author='TylerTemp',
    author_email='tylertempdev@gmail.com',
    description='controllable roop',
    license='MIT',
    keywords='deepfake',
    # url='',
    py_modules=['roopiy'],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Utilities',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
    ],
    install_requires=[
        # 'opencv-python',
        # 'onnx',
        # 'insightface',
        # # 'onnxruntime',
        # 'numpy',
        # 'gfpgan',
        # 'docpie',
        # 'tqdm'
        # # 'torch',
        # # 'torchvision',
    ],
    tests_require=[],
    cmdclass={},
    entry_points={
        'console_scripts': [
            'roopiy = roopiy.__main__:main'
        ]
    },
)
