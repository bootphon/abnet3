# ABnet3 cli documentation

This file explains how to use the ABNet3 cli `abnet3-gridsearch`

The abnet3 cli is designed to take all its parameters as yaml file.
You can see examples of such yaml files in test/data/buckeye.yaml and
test/data/test_embedding.yaml.

## Run one training

To run one training, you must create your yaml file containing the training
parameters, and then run the `abnet3-gridsearch parameters.yaml` command.

The yaml file is designed as following:

```yaml
default_params:
  pathname_experience: path/to/experience
  features:
    class: FeaturesGenerator
    arguments:
      run: never
      ...
  dataloader:
      class: OriginalDataLoader
      arguments:
  sampler:
    class: SamplerClusterSiamese
    arguments:
      run: always
      seed: 0
  model:
    class: SiameseNetwork
    arguments:
      input_dim: 280
  loss:
    class: coscos2
    arguments:
  trainer:
    class: TrainerSiamese
    arguments:
  embedder:
    class: EmbedderSiamese
    arguments:
```

You must define all the arguments for the `features`, `dataloader`, `sampler`,
`model`, `trainer`, `loss` and `embedder` classes (and their class name as
well).

### Inputs and outputs

- The wav files have to be defined in `features.arguments.files`. This is
not mandatory if the features are already generated.
- You can define the path to the output features
in `features.arguments.output_path`
- The cluster file has to be defined in the sampler params
- The input features of the dataloader will automatically be defined
as the output of the features generator. But this can be overridden in the
arguments
- As well, the input to the embedding will automatically be defined as the output
features of the features generator. This can also be overridden.


### features.run and sampler.run

The `run` arguments are special : they allow you to control the feature
generation or the sampling.

You can use `never`, `always`, `once` or `if_none`.

## Run a gridsearch over a certain parameter

You can add at the end of the yaml file the following portion :

```yaml
grid_params:
    sampler:
        arguments:
          type_sampling_mode: ['log','fcube','f','f2','1']
```

This will run a gridsearch over the parameter `type_sampling_mode` of the
sampler.
The gridsearch can only loop over one argument. If you put several arguments
in the grid_params, they will be launched one by one.

