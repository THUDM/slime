import sys
from pathlib import Path

SLIME_ROOT = Path(__file__).resolve().parents[2]
if str(SLIME_ROOT) not in sys.path:
    sys.path.insert(0, str(SLIME_ROOT))

from train import train

from slime.backends.megatron_utils.topk_opd_actor import TopKOPDMegatronTrainRayActor
from slime.ray.actor_group import set_train_actor_cls
from slime.utils.arguments import parse_args


def main():
    args = parse_args()
    if not args.topk_level_opd:
        raise ValueError("topkopd_train.py requires --topk-level-opd.")
    if args.train_backend != "megatron" or args.opd_type != "megatron":
        raise ValueError("topkopd_train.py requires Megatron OPD: --train-backend megatron --opd-type megatron.")
    set_train_actor_cls(TopKOPDMegatronTrainRayActor)
    train(args)


if __name__ == "__main__":
    main()
