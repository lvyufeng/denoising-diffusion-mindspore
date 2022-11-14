from setuptools import setup, find_packages

setup(
  name = 'denoising-diffusion-mindspore',
  packages = find_packages(),
  version = '0.0.1',
  license='MIT',
  description = 'Denoising Diffusion Probabilistic Models - MindSpore',
  author = 'Lyu Yufeng',
  author_email = 'lvyufeng@cqu.edu.cn',
  url = 'https://github.com/lvyufeng/denoising-diffusion-mindspore',
  long_description_content_type = 'text/markdown',
  keywords = [
    'artificial intelligence',
    'generative models',
    'diffusion models',
    'mindspore'
  ],
  install_requires=[
    'mindspore',
    'tqdm'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
  ],
)