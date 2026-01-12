import ray
from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger


def generate_and_save_test_data(args, output_path: str):
    """Generate rollout data and save it for testing."""

    configure_logger()
    pgs = create_placement_groups(args)

    rollout_manager, _ = create_rollout_manager(args, pgs["rollout"])

    actor_model, _ = create_training_models(args, pgs, rollout_manager)

    _ = ray.get(rollout_manager.generate.remote(rollout_id=0))

    ray.get(rollout_manager.dispose.remote())

    return output_path


if __name__ == "__main__":
    args = parse_args()
    output_path = args.save_debug_rollout_data.format(rollout_id=0)
    generate_and_save_test_data(args, output_path)
