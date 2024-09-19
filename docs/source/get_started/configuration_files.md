# Configuration Files Overview

The files under [project-kirby/configs/](https://github.com/nerdslab/project-kirby/blob/main/configs/) contain all the configurations related to data input, data output, hyperparameters, data preprocessing parameters, model parameters, and more.
These files are consumed by Snakefiles and python scripts.
They are orgnaized as below.

# Train Configuration
The `train.yaml` is the parent configuration file that composes the below configuration files. It is used by the training script [`project-kirby/train.py`](https://github.com/nerdslab/project-kirby/blob/main/train.py) to specify all the parametrized information to the model.

# Environment Configuration

The environment configuration files are located in the [`project-kirby/configs/environments/`](https://github.com/nerdslab/project-kirby/blob/main/configs/environments/) directory. These files define the Data Input/Output locations to process data from/to.
Each file corresponds to a different environment and contains paths for temporary, raw, processed, compressed, and uncompressed data.

Based on the location of the data in your local/cluster setup, you can create a new environment file and set the paths accordingly.

These configuration files are used by Snakefile scripts that need to know where to read from or write data to call the right order of python scripts, such as `prepare_data.py`.

# Model Configuration

The model configuration files are located in the [`project-kirby/configs/model/`](https://github.com/nerdslab/project-kirby/blob/main/configs/model/) directory. These files define the hyperparameters for different models used in the project.
Currently we have the following models:
- `poyo-single-session`: smallest, capable for single session data
- `poyo-1`:  More Latents to encode signals, larger embedding dimensions, deeper layers for multi-session data. Model used for NeurIPS 2023 submission ([paper](https://ar5iv.labs.arxiv.org/html/2310.16046)).
- `poyo-galaxy`: Biggest, most capable of decoding multi-session data

# Train & Validation Datasets Configuration

The train datasets configuration files are located in the `project-kirby/configs/train_datasets/` directory.
They define the specifics from the dataset needed for training the model - like the subset of behavior metrics needed for supervision (if multiple behaviors are available), or the subset of sortsets to use for training (if multiple sessions/sortsets are available).

Let's take the example of the [`project-kirby/configs/training_dataset/churchland_single_session.yaml`](https://github.com/nerdslab/project-kirby/blob/main/configs/train_datasets/churchland_single_session.yaml) file to understand the structure of the config file.
```yaml
- selection:
    dandiset: "churchland_shenoy"
    sortset: "nitschke_20090812"
  metrics:
    - output_key: CURSORVELOCITY2D
      loss: True
      weight: 1
  exclude_input:
    - UTAH_ARRAY_WAVEFORMS
```

- The dataset from `churchland_shenoy` set of recordings (published in [Dandi](https://dandiarchive.org/dandiset/000070)) were used to use for training.
- All the recording data corredpondig to the subject `nitschke` will be used for training. If the key-value pairs for "subject" are not specified, all the subjects available under that that dataset's processed folder directory will be used.
- The session (or "sortset") `nitschke_20090812` to use for training `churchland_shenoy` data. If the key-value pairs for "sortset" are not specified, all the sessions available under that that dataset's processed folder directory with be used.
- Note: one can also specify a list of sortsets using "sortsets" key, and "exclude_sortsets" to exclude some sessions from the training :
For example, to include all the sessions except `nitschke_20090812`
    ```yaml
    - selection:
        dandiset: "churchland_shenoy"
        subject: "nitschke"
        exclude_sortsets:
            - "nitschke_20090812"
    ```
    and, to include only sessions `nitschke_20090812` and `nitschke_20090813` for training:
    ```yaml
    - selection:
        dandiset: "churchland_shenoy"
        subject: "nitschke"
        sortsets:
            - "nitschke_20090812"
            - "nitschke_20090813"
    ```

- The `metrics` key is used to specify what behavior metric is to be used for calculating the loss.
- The output behavior data used for supervision in our example is `CURSORVELOCITY2D`. This is one of the types of outputs the model can be trained on. One can view the different outputs available in the [`project-kirby/kirby/taxonomy/taxonomy.py`](https://github.com/nerdslab/project-kirby/blob/main/kirby/taxonomy/taxonomy.py) file.
- `exclude_input` key is used to specify the subset of units to exclude from training, defined by a particular field. In our case, `UTAH_ARRAY_WAVEFORMS` is a source of spike signals that we want to exclude from our training data. More info on the different fields used to characterize input can be found in the [`project-kirby/kirby/taxonomy/taxonomy.py`](https://github.com/nerdslab/project-kirby/blob/main/kirby/taxonomy/taxonomy.py) file.