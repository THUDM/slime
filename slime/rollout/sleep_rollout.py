import time


def sleep(args, rollout_id, data_source, evaluation=False):
    count = 0
    while True:
        time.sleep(3600)
        count += 1
        print(f"rollout sleep for {count} hours")
