# Problem you may face and how to solve

## mask2coco.py
* **RecursionError: maximum recursion depth exceeded in comparison**
    * Go to your conda environment imantics package annotation.py, and change the code:
    ```python
    def index(self, image):

        annotation_index = image.annotations
        category_index = image.categories

        if self.id < 1:
            self.id = len(annotation_index) + 1

        # Increment index until not found
        if annotation_index.get(self.id):
            self.id = max(annotation_index.keys()) + 1

        annotation_index[self.id] = self

        # Category indexing should be case insenstive
        category_name = self.category.name.lower()

        # Check if category exists
        category_found = category_index.get(category_name)
        if category_found:
            # Update category
            self.category = category_found
        else:
            # Index category
            category_index[category_name] = self.category
    ```
## mmdetction
* All the things work previously, but suddenly get **ModuleNotFoundError: No module named 'mmcv'**
    * sometimes this happens when your computer didn't get the package from the right path. You can uninstall mmcv and mmdet and reinstall it again by following the step below:
    ```
    pip uninstall mmdet
    pip uninstall mmcv
    rm -rf ~/.local/lib/python3.7/site-packages
    pip install openmim
    mim install mmdet
    cd mmdetection
    python setup.py install
    ```
* **ImportError: libGL.so.1: cannot open shared object file: No such file or directory**
    * `sudo apt-get update && sudo apt-get install libgl1`
* **RuntimeError: CUDA error: out of memory**
    * There are some scenarios when there are large amount of ground truth boxes, which may cause OOM during target assignment. You can set `gpu_assign_thr=N` in the config of assigner thus the assigner will calculate box overlaps through CPU when there are more than N GT boxes.
    *   Set `with_cp=True` in the backbone. This uses the sublinear strategy in PyTorch to reduce GPU memory cost in the backbone.
    * Try mixed precision training using following the examples in `config/fp16`. The loss_scale might need further tuning for different models.