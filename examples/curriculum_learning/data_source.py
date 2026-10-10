import copy
import logging
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


from slime.rollout.data_source import DataSource, RolloutDataSourceWithBuffer
from slime.utils.misc import load_function
from slime.utils.types import Sample


class MultipleWeightedRolloutDataSourceWithBuffer(DataSource):
    def __init__(self, args):
        self.args = args
        self.rollout_step = 0  # Track the current rollout step

        # Set source names FIRST (needed for logging during weight parsing)
        if hasattr(args, "prompt_data_source_names") and args.prompt_data_source_names:
            self.source_names = args.prompt_data_source_names
        else:
            # Fallback: Extract clean names from file paths (e.g., "dapo-math-17k.jsonl" -> "dapo-math-17k")
            self.source_names = [Path(data_path).stem for data_path in args.prompt_data]

        # Parse and setup weight configuration (one per data source)
        # Each can be: constant, lambda, or function path
        self.weight_fns = self._parse_weights(args.data_source_weights)

        # Create a RolloutDataSourceWithBuffer for each data source
        self.sources = []

        for data_path in args.prompt_data:
            # Create a copy of args with modified prompt_data
            source_args = copy.copy(args)
            source_args.prompt_data = data_path
            self.sources.append(RolloutDataSourceWithBuffer(source_args))

        # Initialize group_index offsets to 0 for each source
        self.group_index_offsets = [0] * len(self.sources)

    def set_rollout_step(self, rollout_step: int):
        """
        Manually set the current rollout step.

        This is optional - by default, rollout_step auto-increments with each get_samples() call.
        Use this only if you need to manually override the step counter (e.g., for testing or
        specific curriculum schedules).

        Args:
            rollout_step: The rollout step to set
        """
        self.rollout_step = rollout_step

    def _parse_weights(self, weights_config: list) -> list[Callable[[int], float]]:
        """
        Parse weight configuration into a list of callable functions, one per data source.

        Each element in weights_config can be:
        1. Float/int constant: 0.5 -> returns constant weight
        2. Lambda string: "lambda step: 0.5 + step/1000" -> returns dynamic weight
        3. Function path: "examples.curriculum_learning.weight_scheduler.source1_weight" -> returns dynamic weight

        Returns a list of functions, each taking rollout_step and returning a single float weight.
        """
        if not isinstance(weights_config, list):
            raise ValueError(f"data_source_weights must be a list, got {type(weights_config)}")

        weight_functions = []
        for i, config in enumerate(weights_config):
            weight_fn = self._parse_single_weight(config, i)
            weight_functions.append(weight_fn)

        logger.info(f"Parsed {len(weight_functions)} weight functions for data sources")
        return weight_functions

    def _parse_single_weight(self, config: float | int | str, index: int) -> Callable[[int], float]:
        """
        Parse a single weight configuration into a callable function.

        Args:
            config: Can be a number (constant), lambda string, or function path
            index: Index of the data source (for logging)

        Returns:
            A function that takes rollout_step and returns a float weight
        """
        # Handle numeric constant
        if isinstance(config, (float, int)):
            logger.info(f"Data source {index} ({self.source_names[index]}): constant weight = {config}")
            return lambda step: float(config)

        # Handle string (lambda or function path)
        elif isinstance(config, str):
            # Check if it's a lambda function
            if config.strip().startswith("lambda"):
                try:
                    weight_fn = eval(config)
                    logger.info(f"Data source {index} ({self.source_names[index]}): lambda function = {config}")
                    return weight_fn
                except Exception as e:
                    logger.error(f"Failed to parse lambda function for source {index}: '{config}': {e}")
                    raise ValueError(f"Invalid lambda function for data source {index}: {config}") from e
            else:
                # Assume it's a function path
                try:
                    weight_fn = load_function(config)
                    logger.info(f"Data source {index} ({self.source_names[index]}): loaded function from {config}")
                    return weight_fn
                except Exception as e:
                    logger.error(f"Failed to load weight function for source {index} from '{config}': {e}")
                    raise ValueError(f"Invalid function path for data source {index}: {config}") from e
        else:
            raise ValueError(f"Weight config for source {index} must be a number or string, got {type(config)}")

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Get samples from data sources according to current weights.

        The rollout step is automatically incremented with each call to track
        curriculum learning progression.

        Args:
            num_samples: Number of sample groups to fetch

        Returns:
            List of sample groups with metadata tagged
        """
        # Compute current weights for this rollout step
        current_weights = [weight_fn(self.rollout_step) for weight_fn in self.weight_fns]

        # Distribute samples according to weights at current rollout step
        samples_per_source = self._distribute_samples_with_weights(num_samples, current_weights)

        samples = []

        for source_idx, (source, n_samples) in enumerate(zip(self.sources, samples_per_source, strict=True)):
            if n_samples > 0:
                source_samples = source.get_samples(n_samples)

                # Apply group_index offset and tag each sample with its source name and weight
                for group in source_samples:
                    for sample in group:
                        # Offset the group_index to make it globally unique across all sources
                        sample.group_index += self.group_index_offsets[source_idx]

                        # Tag with source metadata including the weight used
                        if sample.metadata is None:
                            sample.metadata = {}
                        sample.metadata["data_source"] = self.source_names[source_idx]
                        sample.metadata["data_source_weight"] = current_weights[source_idx]
                        sample.metadata["data_source_rollout_step"] = self.rollout_step

                samples.extend(source_samples)

                # Update offset for next call
                self.group_index_offsets[source_idx] += n_samples

        # Auto-increment rollout step after getting samples
        self.rollout_step += 1

        return samples

    def _distribute_samples_with_weights(self, num_samples: int, weights: list[float]) -> list[int]:
        """
        Distribute num_samples across sources according to provided weights.

        Args:
            num_samples: Number of sample groups to distribute
            weights: List of weights for each data source

        Returns:
            List of sample counts for each source
        """
        if len(weights) != len(self.sources):
            raise ValueError(
                f"Got {len(weights)} weights but there are {len(self.sources)} data sources. " f"Weights: {weights}"
            )

        total = sum(weights)
        if total <= 0:
            raise ValueError(f"Sum of weights must be positive, got {total}. Weights: {weights}")

        normalized = [w / total for w in weights]

        result = []
        remaining = num_samples
        for weight in normalized[:-1]:
            n = round(num_samples * weight)
            result.append(n)
            remaining -= n
        result.append(remaining)  # Give remainder to last source

        return result

    def add_samples(self, samples: list[list[Sample]]):
        # Route samples back to their original source based on metadata
        samples_by_source = [[] for _ in self.sources]

        for group in samples:
            if not group:
                continue
            # Get source name from first sample in group (all should be same source)
            if group[0].metadata and "data_source" in group[0].metadata:
                source_name = group[0].metadata["data_source"]
                # Find the index of this source name
                try:
                    source_idx = self.source_names.index(source_name)
                except ValueError:
                    logger.warning(f"Unknown data source: {source_name}, defaulting to source 0")
                    source_idx = 0
            else:
                source_idx = 0
            samples_by_source[source_idx].append(group)

        # Add to respective source buffers
        for source_idx, source_samples in enumerate(samples_by_source):
            if source_samples:
                self.sources[source_idx].add_samples(source_samples)

    def save(self, rollout_id):
        # Save current rollout_step to checkpoint
        # rollout_step auto-increments with each get_samples() call

        # Save each source with its name
        for i, source in enumerate(self.sources):
            source.save(f"{rollout_id}_{self.source_names[i]}")

        # Save the group_index offsets and rollout_step for resuming from checkpoint
        import torch

        state_dict = {
            "group_index_offsets": self.group_index_offsets,
            "source_names": self.source_names,
            "rollout_step": self.rollout_step,
        }
        save_dir = Path(self.sources[0].args.save) if hasattr(self.sources[0].args, "save") else Path(".")
        offset_path = save_dir / f"multi_source_offsets_{rollout_id}.pt"
        offset_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, offset_path)
        logger.info(
            f"Saved group_index offsets and rollout_step to {offset_path}: offsets={self.group_index_offsets}, step={self.rollout_step}"
        )

    def load(self, rollout_id=None):
        # Load each source with its name
        for i, source in enumerate(self.sources):
            source.load(f"{rollout_id}_{self.source_names[i]}" if rollout_id is not None else None)

        # Load the group_index offsets and rollout_step if they exist
        if rollout_id is not None:
            import torch

            save_dir = Path(self.sources[0].args.save) if hasattr(self.sources[0].args, "save") else Path(".")
            offset_path = save_dir / f"multi_source_offsets_{rollout_id}.pt"
            if offset_path.exists():
                state_dict = torch.load(offset_path, weights_only=True)
                self.group_index_offsets = state_dict["group_index_offsets"]
                self.rollout_step = state_dict.get("rollout_step", 0)  # Fallback to 0 if not in checkpoint
                logger.info(
                    f"Loaded group_index offsets and rollout_step from {offset_path}: offsets={self.group_index_offsets}, step={self.rollout_step}"
                )
            else:
                logger.warning(
                    f"Offset file {offset_path} not found, keeping initialized offsets and rollout_step={self.rollout_step}"
                )
                # Don't modify rollout_step - keep the initialized value (0)

    def __len__(self) -> int:
        return sum(len(source) for source in self.sources)

    def get_buffer_length(self):
        return sum(source.get_buffer_length() for source in self.sources)
