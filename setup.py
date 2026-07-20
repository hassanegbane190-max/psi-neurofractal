from setuptools import setup, find_packages

setup(
    name="psi_neurofractal",
    version="1.0.0",
    author="Gbane Assane",
    author_email="hassanegbane190@gmail.com",
    description="Ψ-NeuroFractal : Detection precoce de maladies neurodegeneratives via reseaux de neurones complexes",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/hassanegbane190-max/psi-neurofractal",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
    ],
    python_requires=">=3.8",
)
