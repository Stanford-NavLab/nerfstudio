"""
Organize images for PolarTraverse dataset

"""

import os
import shutil

if __name__ == "__main__":
    """
    The raw dataset is organized into folders with the following structure:
    ```
    View1/
        Traverse1/
            00m/
                loc0_camL_001ms.png
                loc0_camL_005ms.png
                ...
                loc1_camR_001ms.png
                ...
            01m/
                loc2_camL_001ms.png
                loc2_camL_005ms.png
                ...
                loc3_camR_001ms.png
                ...
            02m/
            ...
            10m/
            poses/
                approx_pose.txt
                refined_pose.txt
        Traverse2/
            00m/
                loc22_camL_001ms.png
                loc22_camL_005ms.png
                ...
                loc23_camR_001ms.png
                ...
            01m/
            ...
            10m/
            poses/
        ...
        Traverse6/
    View2/
    ...
    View4/
    ```

    Each View uses different lighting. The images are named according to the exposure time



    The goal is to organize the images into the following structure:
    ```

    
    """