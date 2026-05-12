import importlib

import modal

app = modal.App("causal-steering")

weights_volume = modal.Volume.from_name("causal-steering-weights", create_if_missing=True)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_pyproject("pyproject.toml")
)


@app.function(
    gpu="A100",
    image=gpu_image,
    volumes={"/weights": weights_volume},
    timeout=3600,
)
def run_remote(fn_path: str, *args, **kwargs):
    """
    Run any causal_steering function on A100.
    fn_path: dotted path, e.g. "causal_steering.steering.loop.run_steering_loop".
    """
    module_path, func_name = fn_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)(*args, **kwargs)
