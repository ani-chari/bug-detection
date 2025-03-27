# setup.py
from setuptools import setup, find_packages

setup(
    name="sima",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.20.0",
        "pillow>=9.0.0",
        "opencv-python>=4.5.0",
        "mss>=6.1.0",
        "pyautogui>=0.9.53",
        "pynput>=1.7.6"
    ],
)
