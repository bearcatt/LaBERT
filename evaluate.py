import argparse
import json

import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

from utils.logger import setup_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate")
    parser.add_argument("--gt_caption", type=str)
    parser.add_argument("--pd_caption", type=str)
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()

    logger = setup_logger("evaluate", args.save_dir, 0)
    ptb_tokenizer = PTBTokenizer()

    scorers = [(Cider(), "C"), (Spice(), "S"),
               (Bleu(4), ["B1", "B2", "B3", "B4"]),
               (Meteor(), "M"), (Rouge(), "R")]

    logger.info(f"loading ground-truths from {args.gt_caption}")
    with open(args.gt_caption) as f:
        gt_captions = json.load(f)
    gt_captions = ptb_tokenizer.tokenize(gt_captions)

    logger.info(f"loading predictions from {args.pd_caption}")
    with open(args.pd_caption) as f:
        pred_dict = json.load(f)
    pd_captions = dict()
    for level, v in pred_dict.items():
        pd_captions[level] = ptb_tokenizer.tokenize(v)

    logger.info("Start evaluating")
    score_all_level = list()
    for level, v in pd_captions.items():
        scores = {}
        for (scorer, method) in scorers:
            score, score_list = scorer.compute_score(gt_captions, v)
            if type(score) == list:
                for m, s in zip(method, score):
                    scores[m] = s
            else:
                scores[method] = score
            if method == "C":
                score_all_level.append(np.asarray(score_list))

        logger.info(
            ' '.join([
                "C: {C:.4f}", "S: {S:.4f}",
                "M: {M:.4f}", "R: {R:.4f}",
                "B1: {B1:.4f}", "B2: {B2:.4f}",
                "B3: {B3:.4f}", "B4: {B4:.4f}"
            ]).format(
                C=scores['C'], S=scores['S'],
                M=scores['M'], R=scores['R'],
                B1=scores['B1'], B2=scores['B2'],
                B3=scores['B3'], B4=scores['B4']
            ))

    score_all_level = np.stack(score_all_level, axis=1)
    logger.info(
        '  '.join([
            "4 level ensemble CIDEr: {C4:.4f}",
            "3 level ensemble CIDEr: {C3:.4f}",
            "2 level ensemble CIDEr: {C2:.4f}",
        ]).format(
            C4=score_all_level.max(axis=1).mean(),
            C3=score_all_level[:, :3].max(axis=1).mean(),
            C2=score_all_level[:, :2].max(axis=1).mean(),
        ))
