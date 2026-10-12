# ClimaX inference on PACE ICE

I used AI to help me understand ClimaX and to write and test the inference code. I found a pretrained checkpoint, but I couldn't find a simple inference example that worked directly with it on ICE. I did not train or fine-tune the model.

## Setup

Setting up the environment was difficult at first. Instead of installing everything manually each time, I made a setup script and ran it on a CPU compute node through SLURM.

```bash
sbatch scripts/setup.slurm
```

The setup script clones the [Microsoft ClimaX repository](https://github.com/microsoft/ClimaX), uses commit `6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c`, and installs the environment from its `docker/environment.yml`. It also downloads the [5.625-degree checkpoint](https://huggingface.co/tungnd/climax), which contains pretrained model weights. It verifies the downloaded checkpoint using SHA-256.

## Inference

I used an LLM to help create `scripts/infer.py`, then tested it on ICE. It loads the checkpoint, reads ERA5 data from [WeatherBench 2](https://weatherbench2.readthedocs.io/en/latest/data-guide.html), and runs ClimaX on an A100 GPU.

```bash
sbatch scripts/infer.slurm
```

I used eight time points and three weather variables: 2 m temperature, 10 m east-west wind, and 10 m north-south wind. I compared the output with ERA5 data six hours later. I also compared it with a basic persistence prediction, which assumes the weather stays the same. The script calculates MAE and RMSE, and runs the same input again to check repeatability.

## Results

I reset the project and ran both scripts again on October 11, 2026. Setup job `6192427` and inference job `6192943` both finished with `COMPLETED 0:0`. Inference ran on an NVIDIA A100 80GB PCIe GPU. Repeating the inference gave the same output (`max difference = 0.0`).

For temperature, the ClimaX RMSE was 8.602 K, compared with 2.465 K for persistence. For the two wind variables, ClimaX RMSE was 5.427 and 4.131 m/s; persistence gave 1.486 and 1.769 m/s.

The full numbers for all eight samples are in `evidence/report.json`. The actual SLURM results and logs are in the `evidence` folder.

## Limitations

The model ran, but its outputs were farther from the ERA5 data than the basic persistence prediction. I used pretrained weights without ERA5 fine-tuning and had to assume normalization values and a lead-time setting. This means my test shows that the inference works, but not how accurate ClimaX would be with the proper forecasting setup.
