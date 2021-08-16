def getPartitions(cell_max_min, nPartx1, nPartx2):
    """
    :param cell_max_min: The size of the entire area
    :param nPartx1: How many partitions along the longitude
    :param nPartx2: How many partitions along the latitude
    :return: a list of all partitions
    """
    width = cell_max_min[1] - cell_max_min[0]
    height = cell_max_min[3] - cell_max_min[2]
    cell_max_min_segs = []
    for x in range(nPartx1):
        for y in range(nPartx2):
            seg_i = (cell_max_min[0] + width / nPartx1 * x, cell_max_min[0] + width / nPartx1 * (x + 1), \
                     cell_max_min[2] + height / nPartx2 * y, cell_max_min[2] + height / nPartx2 * (y + 1))
            cell_max_min_segs.append(seg_i)

    return cell_max_min_segs
