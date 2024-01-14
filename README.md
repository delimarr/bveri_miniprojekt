# setup (windows, with GPU (cuda))
- install python 3.11
- `py -m venv bveri`
- `.\bveri\Scripts\activate`
- `python -m install pip --upgrade pip`
- `pip install poetry`
- `cd underwater_imagery`
- `poetry install` (use: `poetry install --no-cache` if you encounter memory error during torch)
- `python tests\cuda_test.py`

# bveri_miniprojekt
https://ml.pages.fhnw.ch/courses/bveri/hs2023/bveri-lectures-hs2023/exercises/mini_projects.html

# reference project paper: segmentation of underwater imagery [12.01.2024]
- paper: https://arxiv.org/pdf/2004.01241.pdf
- data set: https://irvlab.cs.umn.edu/resources/suim-dataset
- code: https://github.com/xahidbuffon/SUIM-Net
- project page: https://irvlab.cs.umn.edu/image-segmentation/suim
