import copy
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from slime.rollout.data_source import DataSource, RolloutDataSourceWithBuffer
from slime.utils.types import Sample


class MultipleWeightedRolloutDataSourceWithBuffer(DataSource):
    def __init__(self, args):
        self.args = args
        self.weights = args.data_source_weights

        # Create a RolloutDataSourceWithBuffer for each data source
        self.sources = []

        # Use explicit source names from args if provided, otherwise derive from file paths
        if hasattr(args, 'prompt_data_source_names') and args.prompt_data_source_names:
            self.source_names = args.prompt_data_source_names
        else:
            # Fallback: Extract clean names from file paths (e.g., "dapo-math-17k.jsonl" -> "dapo-math-17k")
            self.source_names = [Path(data_path).stem for data_path in args.prompt_data]

        for data_path in args.prompt_data:
            # Create a copy of args with modified prompt_data
            source_args = copy.copy(args)
            source_args.prompt_data = data_path
            self.sources.append(RolloutDataSourceWithBuffer(source_args))

        # Initialize group_index offsets to 0 for each source
        self.group_index_offsets = [0] * len(self.sources)
    
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        # Distribute samples according to weights
        samples_per_source = self._distribute_samples(num_samples)

        samples = []

        for source_idx, (source, n_samples) in enumerate(zip(self.sources, samples_per_source)):
            if n_samples > 0:
                source_samples = source.get_samples(n_samples)

                # Apply group_index offset and tag each sample with its source name
                for group in source_samples:
                    for sample in group:
                        # Offset the group_index to make it globally unique across all sources
                        sample.group_index += self.group_index_offsets[source_idx]

                        # Tag with source metadata
                        if sample.metadata is None:
                            sample.metadata = {}
                        sample.metadata['data_source'] = self.source_names[source_idx]

                samples.extend(source_samples)

                # Update offset for next call
                self.group_index_offsets[source_idx] += n_samples

        return samples
    
    def _distribute_samples(self, num_samples: int) -> list[int]:
        """Distribute num_samples across sources according to weights."""
        total = sum(self.weights)
        normalized = [w / total for w in self.weights]
        
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
            if group[0].metadata and 'data_source' in group[0].metadata:
                source_name = group[0].metadata['data_source']
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
        # Save each source with its name
        for i, source in enumerate(self.sources):
            source.save(f"{rollout_id}_{self.source_names[i]}")

        # Save the group_index offsets for resuming from checkpoint
        import torch
        state_dict = {
            'group_index_offsets': self.group_index_offsets,
            'source_names': self.source_names,
        }
        save_dir = Path(self.sources[0].args.save) if hasattr(self.sources[0].args, 'save') else Path('.')
        offset_path = save_dir / f"multi_source_offsets_{rollout_id}.pt"
        offset_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, offset_path)
        logger.info(f"Saved group_index offsets to {offset_path}: {self.group_index_offsets}")

    def load(self, rollout_id=None):
        # Load each source with its name
        for i, source in enumerate(self.sources):
            source.load(f"{rollout_id}_{self.source_names[i]}" if rollout_id is not None else None)

        # Load the group_index offsets if they exist
        if rollout_id is not None:
            import torch
            save_dir = Path(self.sources[0].args.save) if hasattr(self.sources[0].args, 'save') else Path('.')
            offset_path = save_dir / f"multi_source_offsets_{rollout_id}.pt"
            if offset_path.exists():
                state_dict = torch.load(offset_path, weights_only=True)
                self.group_index_offsets = state_dict['group_index_offsets']
                logger.info(f"Loaded group_index offsets from {offset_path}: {self.group_index_offsets}")
            else:
                logger.warning(f"Offset file {offset_path} not found, keeping initialized offsets")
    
    def __len__(self) -> int:
        return sum(len(source) for source in self.sources)
    
    def get_buffer_length(self):
        return sum(source.get_buffer_length() for source in self.sources)
