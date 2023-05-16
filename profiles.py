import numpy as np


def get_profile(label, labels, stats):
    x, y, w, h, _ = stats[label]
    zone_mask = labels[y:y + h, x:x + w] == label
    left_part, right_part, upper_part, down_part = 0, 0, 0, 0

    for i in range(h):
        idx = 0
        while idx < w and not zone_mask[i, idx]:
            idx += 1
        left_part += idx

        idx = w - 1
        while idx >= 0 and not zone_mask[i, idx]:
            idx -= 1
        right_part += w - 1 - idx

    for i in range(w):
        idx = 0
        while idx < h and not zone_mask[idx, i]:
            idx += 1
        down_part += idx

        idx = h - 1
        while idx >= 0 and not zone_mask[idx, i]:
            idx -= 1
        upper_part += h - 1 - idx

    left_part /= w * h
    right_part /= w * h
    down_part /= w * h
    upper_part /= w * h

    return left_part, right_part, upper_part, down_part


# stats = [0, (0, 0, 3, 5, "shit")]
# labels = np.array([[0, 0, 1],
#                    [0, 1, 1],
#                    [1, 0, 1],
#                    [0, 0, 1],
#                    [0, 0, 1]])
#
# print(get_profile(1, labels, stats))