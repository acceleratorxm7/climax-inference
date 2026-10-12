# ClimaX Inference on PACE ICE

I used AI to help me understand ClimaX and write and test the inference script. I found a pretrained checkpoint, but not a simple example that I could run directly on ICE. My goal was to run inference with an existing model, not train or fine-tune one.

## Setup

Setting up the environment took some trial and error. I had trouble installing everything on the login node, so I made a script to handle the installation in one step on a CPU compute node.

On PACE ICE, clone this repository into `~/climax-inference` and submit the setup job:

```bash
git clone https://github.com/acceleratorxm7/climax-inference.git ~/climax-inference
cd ~/climax-inference
sbatch scripts/setup.slurm
```

The setup script clones [Microsoft ClimaX](https://github.com/microsoft/ClimaX) at commit `6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c`, creates the Python environment using the upstream `docker/environment.yml`, and downloads the [5.625-degree checkpoint](https://huggingface.co/tungnd/climax). The checkpoint contains pretrained model weights, not the weather dataset. The script also checks the checkpoint's SHA-256 hash.

Wait for the job to finish before running inference. Replace `JOB_ID` with the number returned by `sbatch`:

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode
```

The job should show `COMPLETED` with exit code `0:0`.

## Inference

I used an LLM to help create `scripts/infer.py` and debugged it using the ICE job logs. The script loads the checkpoint into ClimaX and reads ERA5 data from [WeatherBench 2](https://weatherbench2.readthedocs.io/en/latest/data-guide.html). It runs the model on an A100 GPU.

```bash
sbatch scripts/infer.slurm
```

After the job finishes, check its status and read the output:

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode
cat results/era5/report.json
```

I tested eight time points with three variables: 2 m temperature and the two 10 m wind components. I compared the model output against ERA5 six hours later, as well as a simple baseline that assumes the weather does not change. The script calculates latitude-weighted MAE and RMSE. It also checks that the checkpoint loads, the output shape is correct, there are no NaN or infinite values, and repeated inference gives the same output.

The script saves `results/era5/report.json` and `results/era5/prediction.npz`. A copy of the JSON report from my run is in `evidence/report.json`.

## Results and Evidence

I reset the project and ran setup and inference again on October 11, 2026. Setup job `6192427` completed in 10 minutes 50 seconds on `ice-cpu`. Inference job `6192943` completed in 43 seconds on `ice-gpu`, using an NVIDIA A100 80GB PCIe. Both jobs finished with exit code `0:0`.

The model produced the same output when I repeated inference (`maximum difference = 0.0`). The ClimaX RMSE was 8.602 K for temperature and 5.427 and 4.131 m/s for the two wind components. The simple baseline had lower errors in this test.

The actual results are in `evidence/report.json`. I also included `evidence/slurm.txt` for job statuses, `evidence/setup.log` for installation output, and `evidence/inference.log` for the inference output.

## Limitations

The model ran, but its output was not as close to the ERA5 data as the simple baseline. I used pretrained weights without ERA5 forecast fine-tuning and assumed normalization values and a lead-time setting. So this test shows that I could run and check inference on ICE, not how accurate ClimaX would be with the correct forecasting setup.
