from .train import train, parse_args

if __name__ == "__main__":
    print("***** WARN: This entrypoint is deprecated. Please use train.py instead. *****")
    args = parse_args()
    train(args)
