# Prophet: An LLM Inference Engine Optimized For Head-of-Line Blocking

To start the benchmark, run:
`python run_ray.py`

The default configuration is for running on an AWS p4d.24xlarge instance. To adjust the number of prefiller and decoder GPUs, change `config/coordinators/ray_coordinator.yaml`.

If memory runs out, the batch size can be change in the prefiller and decoder scheduler configurations.


