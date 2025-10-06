import os
from torch.utils.tensorboard import SummaryWriter


class _TensorboardAdapter:
    _instance = None
    _writer = None

    """
    # Usage example: This will return the same instance every rank
    # tb = _TensorboardAdapter(args)  # Initialize on first call
    # tb.log({"Loss": 0.1}, step=1)

    # In other files:
    # from tensorboard_utils import _TensorboardAdapter
    # tb = _TensorboardAdapter(args)  # No parameters needed to get existing instance
    # tb.log({"Accuracy": 0.9}, step=1)
    """

    def __new__(cls, args):
        assert args.use_tensorboard, f"{args.use_tensorboard=}"
        tb_project_name = args.tb_project_name
        tb_experiment_name = args.tb_experiment_name
        if cls._instance is None:
            cls._instance = super(_TensorboardAdapter, cls).__new__(cls)
            # Initialize if parameters are provided during first creation
            if (tb_project_name is not None and tb_experiment_name is not None) or os.environ.get(
                "TENSORBOARD_DIR", None
            ):
                cls._instance._initialize(tb_project_name, tb_experiment_name)
        return cls._instance

    def _initialize(self, tb_project_name, tb_experiment_name):
        """Actual initialization logic"""
        # Get tensorboard directory from environment variable or use default path
        tensorboard_dir = os.environ.get("TENSORBOARD_DIR", f"tensorboard_log/{tb_project_name}/{tb_experiment_name}")
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f"Saving tensorboard log to {tensorboard_dir}.")
        self.writer = SummaryWriter(tensorboard_dir)

    def log(self, data, step):
        """Log data to tensorboard

        Args:
            data (dict): Dictionary containing metric names and values
            step (int): Current step/epoch number
        """
        for key in data:
            self.writer.add_scalar(key, data[key], step)

    def finish(self):
        """Close the tensorboard writer"""
        self.writer.close()
