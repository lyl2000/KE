import random


def shuffle(lol, seed):
    """shuffle inplace each list in the same order

    Args:
        lol: list of list as input
        seed: seed the shuffling
    """
    for l in lol:
        random.seed(seed)
        random.shuffle(l)


def contextwin(l, win):
    """given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding to context windows surrounding each word in the sentence

    Args:
        l: a list of indexes composing a sentence
        win (int): corresponding to the size of the window

    Returns:
        a list of list of indexes corresponding to context windows surrounding each word in the sentence
    """
    assert win % 2 == 1 and win >= 1
    l = list(l)

    lpadded = win // 2 * [0] + l + win // 2 * [0]
    out = [lpadded[i: i+win] for i in range(len(l))]

    assert len(out) == len(l)
    return out


def contextwin_2(ls, win):
    assert win % 2 == 1 and win >= 1
    outs = []
    for l in ls:
        outs.append(contextwin(l, win))
    return outs


def getKeyphraseList(l):
    res, now = [], []
    for i in range(len(l)):
        if l[i] != 0:
            now.append(str(i))
        if l[i] == 0 or i == len(l) - 1:
            if len(now) != 0:
                res.append(' '.join(now))
            now = []
    return set(res)


def conlleval(predictions, groundtruth, file):
    assert len(predictions) == len(groundtruth)
    res = {}
    all_cnt, good_cnt = len(predictions), 0
    p_cnt, r_cnt, pr_cnt = 0, 0, 0
    for i in range(all_cnt):
        # print i
        if all(predictions[i][:len(groundtruth[i])] == groundtruth[i]):
            good_cnt += 1
        pKeyphraseList = getKeyphraseList(predictions[i][:len(groundtruth[i])])
        gKeyphraseList = getKeyphraseList(groundtruth[i])
        p_cnt += len(pKeyphraseList)
        r_cnt += len(gKeyphraseList)
        pr_cnt += len(pKeyphraseList & gKeyphraseList)
    print(good_cnt, all_cnt, p_cnt, r_cnt)
    res['a'] = good_cnt / all_cnt if all_cnt > 0 else 0
    res['p'] = good_cnt / p_cnt if p_cnt > 0 else 0
    res['r'] = good_cnt / r_cnt if r_cnt > 0 else 0
    res['f'] = 2 * res['p'] * res['r'] / (res['p'] + res['r']) if res['p'] + res['r'] > 0 else 0
    return res
