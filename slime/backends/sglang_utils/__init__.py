from slime.utils import accelerator

# Finalize the backend before importing any SGLang submodule. In a MUSA
# runtime this loads musa_patch only after MUSA wins backend selection.
accelerator.initialize_accelerator()
