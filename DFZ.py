import numpy as np


def get_DFZ(labels, label, stats, num_zones=4):
    x, y, w, h, _ = stats[label]

    densities = []
    i_step = w // num_zones
    j_step = h // num_zones
    for i in range(num_zones):
        for j in range(num_zones):
            x1, y1, x2, y2 = x + i * i_step, y + j * j_step, x + (i + 1) * i_step, y + (j + 1) * j_step
            zone_mask = labels[y1:y2, x1:x2] == label
            zone_size = np.count_nonzero(zone_mask)
            if (y2 - y1) * (x2 - x1) == 0 or np.isnan(zone_size):
                zone_density = 0
            else:
                zone_density = zone_size / ((y2 - y1) * (x2 - x1))
            densities.append(zone_density)

    return densities
