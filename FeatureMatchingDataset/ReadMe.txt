The images are acquired for the purpose of testing object detection using feature mathcing.
An example of the feature matching algorithm can be seen here:
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html

There's a possibility that this will not work at all, in any case also, also trying out another approach is encouraged.

The images can be used for varyious level of evaluation. In order of increasing difficulty:
-A component image, which is envisioned to be the "training" part.
-Images with only a single part for simple detection.
-Images with 3 different parts.
-Images with two similar parts.
-Images with all 6 parts nicely seperated
-Images with all 6 parts close together

Images which only contains a subset of the parts can be tested both with detection where all parts are possible or where only the present parts can be detected.
