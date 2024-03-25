import subprocess
import time
import contextlib
from tritonclient.http import InferenceServerClient

# Define image https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
tag = 'nvcr.io/nvidia/tritonserver:23.09-py3'  # 6.4 GB

# Pull the image
subprocess.call(f'docker pull {tag}', shell=True)

# Run the Triton server and capture the container ID
triton_repo_path = "/home/loki/PycharmProjects/counting/tmp/triton_repo"
container_id = subprocess.check_output(
    f'docker run -d --rm -v {triton_repo_path}:/models -p 7000:7000 {tag} tritonserver --model-repository=/models',
    shell=True).decode('utf-8').strip()

# Wait for the Triton server to start
triton_client = InferenceServerClient(url='localhost:8000', verbose=False, ssl=False)

# Wait until model is ready
for _ in range(10):
    with contextlib.suppress(Exception):
        assert triton_client.is_model_ready(model_name)
        break
    time.sleep(1)