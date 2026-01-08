# Usage

For the minimal demo:

```shell
# install dependencies
apt update && apt install -y docker-cli
pip install kimina-client polars

# prepare data
python examples/formal_math/single_round/prepare_data.py --output-name minimal_demo

# prepare ray, model, test dataset, etc
# normally just use this script, but here we want to demonstrate run_minimal.py, thus skip ray-submit part
SLIME_SCRIPT_ENABLE_RAY_SUBMIT=0 python examples/formal_math/single_round/run.py

# run
python examples/formal_math/single_round/run_minimal.py
```

The code also support more complicated cases, e.g.:

* SFT + RL
* Data filter + RL
