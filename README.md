# promeai/FLUX.1-controlnet-lineart-promeai Cog Model

This is an implementation of the [promeai/FLUX.1-controlnet-lineart-promeai](https://huggingface.co/promeai/FLUX.1-controlnet-lineart-promeai) as a [Cog](https://github.com/replicate/cog) model.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own model to [Replicate](https://replicate.com).


## How to use

Make sure you have [cog](https://github.com/replicate/cog) installed.

To run a prediction:

    cog predict -i prompt="cute anime girl with massive fluffy fennec ears and a big fluffy tail blonde messy long hair blue eyes wearing a maid outfit with a long black gold leaf pattern dress and a white apron mouth open holding a fancy black forest cake with candles on top in the kitchen of an old dark Victorian mansion lit by candlelight with a bright window to the foggy forest and very expensive stuff everywhere" -i control_image=@control.jpg

<div style="display: flex; justify-content: space-between;">
    <img src="control.jpg" alt="Control Image" width="48%">
    <img src="output.png" alt="Output" width="48%">
</div>