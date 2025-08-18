from setuptools import setup, find_packages

setup(
    name="temporal_classification_model",
    version="0.1.0",
    author="David Yang",
    description="動態偵測模型，用於判斷物件的時序動態狀態",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
    ],
)