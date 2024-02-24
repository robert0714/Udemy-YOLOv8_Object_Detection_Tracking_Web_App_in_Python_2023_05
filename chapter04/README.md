# Chapter4 YOLOv8 Object Tracking
* Deep Sort with PyTorch:    
  https://github.com/ZQPei/deep_sort_pytorch
## How we can integrate Deepsort object tracking with YOLO
* github: https://github.com/MuhammadMoinFaisal/YOLOv8-DeepSORT-Object-Tracking
* deepSORT File : https://drive.google.com/drive/folders/1kna8eWGrSfzaR6DtNJ8_GchGgPMv3VC8?usp=sharing
* Demo video:
  * url: https://drive.google.com/file/d/1rjBn8Fl1E_9d0EMVtL24S9aNQOJAveR5
  * the download cli in colab
    ```bash
    gdown "https://drive.google.com/uc?id=1rjBn8Fl1E_9d0EMVtL24S9aNQOJAveR5&confirm=t"
    ```
* For yolov8 object detection + Tracking + Vehicle Counting
  * Google Drive Link
    ```bash
    https://drive.google.com/drive/folders/1awlzTGHBBAn_2pKCkLFADMd1EN_rJETW?usp=sharing
    ```
### Issue: UnicodeDecodeError: 'cp950' codec can't decode byte 0xf0 in position 19: illegal multibyte sequence
* To modify setup.py
  ```python
  # line 13
  REQUIREMENTS = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements((ROOT / 'requirements.txt').read_text())]

  # line 18 
      return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', file.read_text(), re.M)[1]
  ```
  * Specific Character UTF8
  ```python
  # line 13
  REQUIREMENTS = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements((ROOT / 'requirements.txt').read_text(encoding="utf-8"))]

  # line 18 
      return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', file.read_text(encoding="utf-8"), re.M)[1]
  ```
### Issue: WARNING: Ignore distutils configs in setup.cfg due to encoding errors.
* https://github.com/pypa/pip/issues/11066
  
### Dependency issues
```
pip install easydict
```
* AttributeError: module 'numpy' has no attribute 'float'.
  * NumPy 1.20 ([release notes](https://numpy.org/doc/stable/release/1.20.0-notes.html#deprecations)) deprecated numpy.float, numpy.int, and similar aliases, causing them to issue a deprecation warning
  * NumPy 1.24 ([release notes](https://numpy.org/doc/stable/release/1.24.0-notes.html#expired-deprecations)) removed these aliases altogether, causing an error when they are used
    ```
    pip install numpy==1.23.5
    ```
  * reference: https://stackoverflow.com/questions/74844262/how-can-i-solve-error-module-numpy-has-no-attribute-float-in-python
