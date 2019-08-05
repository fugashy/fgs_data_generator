# fsg_data_generator

Generate data file for several purpose.

# Executable file

- generate_2d

    Generate 2d data

    There are several data model and noise model (not yet. will be there near future.)

    ```bash
    ros2 run fgs_data_generator generate __params:=PATH_TO_CONFIG.yaml
    ```

    or

    ```bash
    cd PATH_TO_THIS_PKG/fgs_data_generator
    python3 generate PATH_TO_CONFIG.yaml
    ```

  ![data_sample](https://github.com/fugashy/fgs_opt/blob/images/ellipse_data.png)
