import math


def learning_rate_schedule(time, alpha_max, alpha_min, warmup_iter, num_cosine_iter):
    """
    learning_rate_schedule computes a learning rate required for the training
    depending upon the current stage of training.
    Study : https://www.notion.so/fountain-pen/Learning-rate-Scheduling-28285d554a8d80629bcae7f2fab6b2c5

    Args:
        time - current time 't'
        alpha_max - maximum allowed learning rate
        alpha_min - min allowed learning rate
        warmup_iter - Time until which alpha_max is used and post this, annealing kicks in.
        num_cosine_ter - Number of annealing iterations. The discrete steps which will sample
                        the cosine annealing lr.
    Return
        y_lr - output learning rate
    """
    if time < warmup_iter:
        y_lr = time * alpha_max / warmup_iter
        return y_lr
    if time > num_cosine_iter:
        y_lr = alpha_min
        return y_lr
    alpha_t = alpha_min + 0.5 * (alpha_max - alpha_min) * (
        1 + math.cos((time - warmup_iter) * math.pi / (num_cosine_iter - warmup_iter))
    )
    return alpha_t
