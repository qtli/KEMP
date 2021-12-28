from torch import nn
import nltk

def init_xavier_weight(w):
    nn.init.xavier_normal_(w)

def init_bias(b):
    nn.init.constant_(b, 0.)

def init_linear_weight(linear):
    init_xavier_weight(linear.weight)
    if linear.bias is not None:
        init_bias(linear.bias)

def init_uniform_weight(w):
    nn.init.normal_(w, -0.1, 0.1)


def ave_rouge(all_rouges):
    '''
    [{"rouge-1": {"f": 0.49411764217577864,
              "p": 0.5833333333333334,
              "r": 0.42857142857142855},
  "rouge-2": {"f": 0.23423422957552154,
              "p": 0.3170731707317073,
              "r": 0.18571428571428572},
  "rouge-l": {"f": 0.42751590030718895,
              "p": 0.5277777777777778,
              "r": 0.3877551020408163}}]
    :param all_rouges:
    :return:
    '''
    all_rouge_1 = []
    all_rouge_2 = []
    all_rouge_l = []

    ave_rouge_1 = {}
    ave_rouge_2 = {}
    ave_rouge_l = {}

    for each_r in all_rouges:
        all_rouge_1.append(each_r[0]["rouge-1"])
        all_rouge_2.append(each_r[0]["rouge-2"])
        all_rouge_l.append(each_r[0]["rouge-l"])

    f = 0.0
    p = 0.0
    r = 0.0
    count = 0
    for each in all_rouge_1:
        f += each["f"]
        p += each["p"]
        r += each["r"]
        count += 1
    ave_rouge_1["rouge-1"] = {"f": f/count, "p": p/count, "r": r/count}

    f = 0.0
    p = 0.0
    r = 0.0
    count = 0
    for each in all_rouge_2:
        f += each["f"]
        p += each["p"]
        r += each["r"]
        count += 1
    ave_rouge_2["rouge-2"] = {"f": f / count, "p": p / count, "r": r / count}

    f = 0.0
    p = 0.0
    r = 0.0
    count = 0
    for each in all_rouge_l:
        f += each["f"]
        p += each["p"]
        r += each["r"]
        count += 1
    ave_rouge_l["rouge-l"] = {"f": f / count, "p": p / count, "r": r / count}

    result = []
    result.append(ave_rouge_1)
    result.append(ave_rouge_2)
    result.append(ave_rouge_l)

    return result


def distinctEval(preds):
    response_ugm = set([])
    response_bgm = set([])
    response_len = sum([len(p) for p in preds])  # 参与计算的句子的总词数

    for path in preds:
        for u in path:
            response_ugm.add(u)
        for b in list(nltk.bigrams(path)):  # TODO 中文distinct
            response_bgm.add(b)
    response_len_ave = response_len/len(preds)
    distinctOne = len(response_ugm)/response_len
    distinctTwo = len(response_bgm)/response_len
    return distinctOne, distinctTwo, response_len_ave