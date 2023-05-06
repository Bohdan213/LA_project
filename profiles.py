import numpy as np


def get_profile(label, labels, stats):
    def recursion(i, j):
        if i >= h or j >= w or i < 0 or j < 0 or is_visit[i][j]:
            return 0

        cnt = 0
        is_visit[i][j] = True
        cnt += zone_mask[i][j]
        cnt += recursion(i + 1, j)
        cnt += recursion(i - 1, j)
        cnt += recursion(i, j + 1)
        cnt += recursion(i, j - 1)
        return cnt

    x, y, w, h, _ = stats[label]
    zone_mask = labels[y:y + h, x:x + w] == label
    is_visit = [[False for _ in range(w)] for _ in range(h)]
    left_part, right_part, upper_part, down_part = 0, 0, 0, 0

    for i in range(y, y + h):
        if not zone_mask[i][0]:
            left_part += recursion(i, 0)
            is_visit = [[False for _ in range(w)] for _ in range(h)]

    left_part = left_part / w / h

    for i in range(y + h - 1, y - 1, -1):
        if not zone_mask[i][x + w - 1]:
            right_part += recursion(i, x + w - 1)
            is_visit = [[False for _ in range(w)] for _ in range(h)]

    right_part = right_part / w / h

    for i in range(x, x + w):
        if not zone_mask[0][i]:
            down_part += recursion(0, i)
            is_visit = [[False for _ in range(w)] for _ in range(h)]

    down_part = down_part / w / h

    for i in range(x + w - 1, x - 1, -1):
        if not zone_mask[y + h - 1][i]:
            upper_part += recursion(y + h - 1, i)
            is_visit = [[False for _ in range(w)] for _ in range(h)]

    upper_part = upper_part / w / h

    return left_part, right_part, upper_part, down_part


stats = [0, (0, 0, 3, 5, "shit")]
labels = np.array([[0, 0, 1],
                   [0, 1, 1],
                   [1, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1]])

print(get_profile(1, labels, stats))
