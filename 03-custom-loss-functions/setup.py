from setuptools import setup

setup(
    name="custom-loss-test",
    version="0.1.0",
    description="Test project for PyTrim with custom loss functions",
    author="Kadir Guzel",
    py_modules=['custom_loss_example', 'custom_loss_example_with_issues', 'simple_test'],
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.21.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.5.0",
        "Pillow>=8.3.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "requests>=2.26.0",
        "scikit-learn>=1.0.0",
        "seaborn>=0.11.0",
        "opencv-python>=4.5.0",
        "transformers>=4.15.0",
        "tensorflow>=2.7.0",
        "jax>=0.2.25",
        "plotly>=5.5.0"
    ],
    python_requires=">=3.8",
)