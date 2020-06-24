import sys
import os
import matplotlib.pyplot as plt
import numpy as np

class HilbertCodec(object):
    def __init__(self, level):
        if level < 1:
            raise ValueError("level<1")
        self._level = level
        self._mapping = [
            [0, 1, 3, 2],
            [2, 1, 3, 0],
            [0, 3, 1, 2],
            [2, 3, 1, 0]
        ]
        self._anti_mapping = [
            [0, 1, 3, 2],
            [3, 1, 0, 2],
            [0, 2, 3, 1],
            [3, 2, 0, 1]
        ]

    def label(self):
        return "hilbert"

    def encode(self, coord):
        x, y = coord
        cardinality = 2 ** self._level

        if x >= cardinality or y >= cardinality:
            return -1

        min_x, min_y = [0] * 2
        max_x, max_y = [cardinality - 1] * 2
        cnt_hidx_0, cnt_hidx_3 = [0] * 2

        res = 0
        for i in xrange(self._level):
            mid_x = (min_x + max_x) / 2
            mid_y = (min_y + max_y) / 2

            qidx = 0
            if x <= mid_x and y <= mid_y:
                qidx = 0
                max_x = mid_x
                max_y = mid_y
            elif x <= mid_x and y > mid_y:
                qidx = 1
                max_x = mid_x
                min_y = mid_y + 1
            elif x > mid_x and y <= mid_y:
                qidx = 2
                min_x = mid_x + 1
                max_y = mid_y
            else:
                qidx = 3
                min_x = mid_x + 1
                min_y = mid_y + 1
            mapping_idx = ((cnt_hidx_0 % 2) << 1) | (cnt_hidx_3 % 2)
            hidx = self._mapping[mapping_idx][qidx]
            res = (res << 2) | hidx

            cnt_hidx_0 += (hidx == 0)
            cnt_hidx_3 += (hidx == 3)
        return res

    def decode(self, bits):
        x, y = [0] * 2
        cnt_hidx_0, cnt_hidx_3 = [0] * 2

        offset = 2 * self._level - 2
        for i in xrange(self._level):
            hidx = ((3 << offset) & bits) >> offset
            mapping_idx = ((cnt_hidx_0 % 2) << 1) | (cnt_hidx_3 % 2)
            qidx = self._anti_mapping[mapping_idx][hidx]
            x = ((x << 1) | ((qidx & 2) >> 1))
            y = ((y << 1) | (qidx & 1))
            cnt_hidx_0 += (hidx == 0)
            cnt_hidx_3 += (hidx == 3)
            offset -= 2
        return [x, y]

    def to_range(self, quad):
        p1_x, p1_y = quad[0]
        p2_x, p2_y = quad[1]
        hilbs = [ self.encode(p) for p in [
            [p1_x, p1_y], [p2_x, p2_y], [p1_x, p2_y], [p2_x, p1_y] ]]
        return [min(hilbs), max(hilbs)]

    def to_quad(self, min_, max_):
        bp1 = self.decode(min_)
        bp2 = self.decode(max_)
        size = max(abs(bp2[0] - bp1[0]), abs(bp2[1] - bp1[1]))
        if bp1 < bp2:
            return [[bp1[0], bp1[1] + size], [bp1[0] + size, bp1[1]]]
        else:
            return [[bp1[0] - size, bp1[1]], [bp1[0], bp1[1] - size]]


def contain(b1, b2):
    return (b1[0][0] <= b2[0][0] and b1[0][1] >= b2[0][1]
        and b1[1][0] >= b2[1][0] and b1[1][1] <= b2[1][1])

def intersect(b1, b2):
    return not (b1[0][0] > b2[1][0] or b1[0][1] < b2[1][1]
        or b1[1][0] < b2[0][0] or b1[1][1] > b2[0][1])

def fill_window(window, parent):
    pp1, pp2 = parent
    if pp1[0] == pp2[0]:
        return [parent]
    pc = [(pp1[0] + pp2[0]) / 2, (pp1[1] + pp2[1]) / 2]
    res = []
    for bucket in [[pp1, [pc[0], pc[1] + 1]],
            [[pc[0] + 1, pc[1]], pp2], \
            [[pp1[0], pc[1]], [pc[0], pp2[1]]], \
            [[pc[0] + 1, pp1[1]], [pp2[0], pc[1] + 1]]]:
        if contain(window, bucket):
            res.append(bucket)
        elif intersect(window, bucket):
            res += fill_window(window, bucket)
    return res

class ZCodec(object):
    def __init__(self, level):
        if level < 1:
            raise ValueError("level<1")
        self._level = level

    def label(self):
        return "z-curve"

    def encode(self, coord):
        x, y = coord
        cardinality = 2 ** self._level

        if x >= cardinality or y >= cardinality:
            return -1

        min_x, min_y = [0] * 2
        max_x, max_y = [cardinality - 1] * 2

        res = 0
        for i in xrange(self._level):
            mid_x = (min_x + max_x) / 2
            mid_y = (min_y + max_y) / 2

            qidx = 0
            if x <= mid_x and y <= mid_y:
                qidx = 0
                max_x = mid_x
                max_y = mid_y
            elif x <= mid_x and y > mid_y:
                qidx = 1
                max_x = mid_x
                min_y = mid_y + 1
            elif x > mid_x and y <= mid_y:
                qidx = 2
                min_x = mid_x + 1
                max_y = mid_y
            else:
                qidx = 3
                min_x = mid_x + 1
                min_y = mid_y + 1
            res = (res << 2) | qidx
        return res

    def decode(self, bits):
        x, y = [0] * 2
        cnt_hidx_0, cnt_hidx_3 = [0] * 2

        offset = 2 * self._level - 2
        for i in xrange(self._level):
            qidx = ((3 << offset) & bits) >> offset
            x = ((x << 1) | ((qidx & 2) >> 1))
            y = ((y << 1) | (qidx & 1))
            offset -= 2
        return [x, y]

    def to_range(self, quad):
        p1_x, p1_y = quad[0]
        p2_x, p2_y = quad[1]
        zs = [ self.encode(p) for p in [
            [p1_x, p1_y], [p2_x, p2_y], [p1_x, p2_y], [p2_x, p1_y] ]]
        return [min(zs), max(zs)]

    def to_quad(self, min_, max_):
        bp1 = self.decode(min_)
        bp2 = self.decode(max_)
        return [[bp1[0], bp2[1]], [bp2[0], bp1[1]]]

def contain(b1, b2):
    return (b1[0][0] <= b2[0][0] and b1[0][1] >= b2[0][1]
        and b1[1][0] >= b2[1][0] and b1[1][1] <= b2[1][1])

def intersect(b1, b2):
    return not (b1[0][0] > b2[1][0] or b1[0][1] < b2[1][1]
        or b1[1][0] < b2[0][0] or b1[1][1] > b2[0][1])

def fill_window(window, parent):
    pp1, pp2 = parent
    if pp1[0] == pp2[0]:
        return [parent]
    pc = [(pp1[0] + pp2[0]) / 2, (pp1[1] + pp2[1]) / 2]
    res = []
    for bucket in [[pp1, [pc[0], pc[1] + 1]],
            [[pc[0] + 1, pc[1]], pp2], \
            [[pp1[0], pc[1]], [pc[0], pp2[1]]], \
            [[pc[0] + 1, pp1[1]], [pp2[0], pc[1] + 1]]]:
        if contain(window, bucket):
            res.append(bucket)
        elif intersect(window, bucket):
            res += fill_window(window, bucket)
    return res

def bbox_to_ranges(codec, window):
    min_hilb, max_hilb = codec.to_range(window)
    n = 0
    xor = min_hilb ^ max_hilb
    while xor > 0:
        xor >>= 1
        n += 1
    offset = (n + 1) / 2 * 2
    mask = (1 << offset) - 1
    parent_quad = codec.to_quad(min_hilb & (~mask), min_hilb | mask)
    # merge ranges as possible
    sorted_ranges = sorted(
        [codec.to_range(x) for x in fill_window(window, parent_quad)])
    print >> sys.stderr, "deepest of the parent quadrant: %s" % (parent_quad)
    res = []
    curr_range = sorted_ranges[0]
    for r in sorted_ranges[1:]:
        if curr_range[-1] + 1 == r[0]:
            curr_range[-1] = r[-1]
        else:
            res.append(curr_range)
            curr_range = r
    res.append(curr_range)
    return res

def bbox_decomp(codec, bbox):
    bbox_x_l, bbox_x_r, bbox_y_b, bbox_y_u = \
        bbox[0][0], bbox[1][0], bbox[1][1], bbox[0][1]

    plt.plot([bbox_x_l, bbox_x_l, bbox_x_r, bbox_x_r, bbox_x_l], \
            [bbox_y_b, bbox_y_u, bbox_y_u, bbox_y_b, bbox_y_b], \
            label='bbox', linewidth=3, linestyle='--', color='0.75')
    i = 0
    for b, e in bbox_to_ranges(codec, bbox):
        x_ = []
        y_ = []
        for m in range(b, e + 1):
            pos = codec.decode( m)
            x_.append(pos[0])
            y_.append(pos[1])
            plt.plot(x_, y_, label="%d,%d" % (b, e), linewidth=1.2)
        i += 1
    plt.title("bbox_decomp  decompose bbox to ranges\nnumber of ranges: %s"\
        % i)

def locality1(codec, rg):
    b, e = rg
    x_ = []
    y_ = []
    for m in range(b, e + 1):
        pos = codec.decode(m)
        x_.append(pos[0])
        y_.append(pos[1])
    plt.plot(x_, y_, label="%d,%d" % (b, e), linewidth=1.2)
    plt.title("locality1  area of range\nlen of range: %s" % (e - b + 1))

def locality2(codec, bbox):
    bbox_x_l, bbox_x_r, bbox_y_b, bbox_y_u = \
        bbox[0][0], bbox[1][0], bbox[1][1], bbox[0][1]

    plt.plot([bbox_x_l, bbox_x_l, bbox_x_r, bbox_x_r, bbox_x_l], \
            [bbox_y_b, bbox_y_u, bbox_y_u, bbox_y_b, bbox_y_b], \
            label='bbox', linewidth=3, linestyle='--', color='0.75')

    ranges = bbox_to_ranges(codec, bbox)
    b = min([r[0] for r in ranges])
    e = max([r[1] for r in ranges])

    x_ = []
    y_ = []
    for m in range(b, e + 1):
        pos = codec.decode(m)
        x_.append(pos[0])
        y_.append(pos[1])
    plt.plot(x_, y_, label="%d,%d" % (b, e), linewidth=1.2)
    plt.title("locality2  bbox to range\nlen of range: %s"\
        % (e - b + 1))

def get_codec():
    if len(sys.argv) > 1 and sys.argv[1] == 'hilb':
        return lambda l : HilbertCodec(l)
    else:
        return lambda l : ZCodec(l)

def diagram(level, plot_curvs):
    codec = get_codec()(level)

    dim = 2 ** level
    xy_ticks = np.arange(-0.5, dim + 0.5, 1)

    x = []
    y = []
    for i in xrange(dim ** 2):
        pos = codec.decode(i)
        x.append(pos[0])
        y.append(pos[1])

    for i, curve in enumerate(plot_curvs):
        plt.figure(i + 1)
        plt.xticks(xy_ticks)
        plt.yticks(xy_ticks)
        plt.plot(x, y, label=codec.label(), linewidth=1.2)
        plt.legend()
        plt.grid(True, which='major', linestyle='-')

        curve(codec)
        i += 1

    plt.show()

def main():
    level = 5
    bbox = [[1, 11], [11, 1]]
    rg=[200, 400]
    diagram(level, [ lambda codec : locality1(codec, rg),
                     lambda codec : locality2(codec, bbox),
                     lambda codec : bbox_decomp(codec, bbox) ])

if __name__ == '__main__':
    main()
