def pop_first(buffer, num_samples):
    samples = []
    for _ in range(num_samples):
        if buffer:
            samples.append(buffer.pop(0))
    return samples


def get_newest_samples(buffer, num_samples):
    return buffer[-num_samples:]
