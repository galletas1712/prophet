# Prophet: An LLM Inference Engine Optimized For Head-of-Line Blocking

To start the benchmark, run:
`python run_ray.py`

The default configuration is for running on an AWS p4d.24xlarge instance. To adjust the number of prefiller and decoder GPUs, change `config/coordinators/ray_coordinator.yaml`.

If memory runs out, the batch size can be change in the prefiller and decoder scheduler configurations.

## Running in Docker
In the project directory, build the image using
```
docker build -t prophet .
```

Then, you can run the image in a container using the `start_container.sh` script.

## Profiling
To profile with nsys (without CPU sampling), run the following.
```nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -o nsight_report -f true -x true python benchmarks/decode_ctx_switch_in_gpumem.py```
