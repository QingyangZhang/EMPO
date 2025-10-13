from verl.utils.reward_score.physics.judge import Judger

def compute_score(data_item, precision=1e-4) -> bool:
    """
    Check the correctness for single item. 

    Parameters:
        data_item: Data to be checked. 
            data format:

    Returns:
        correctness: A list that contains the correctness for all prompts. 
    """

    ugphysics_judger = Judger(strict_extract=True)
    correctness, extracted_pred, extracted_gt = ugphysics_judger.auto_judge(data_item['completion'],
                                              data_item['answers'],
                                              precision=precision)
    # if not correctness:
    #     correctness, msg = ugphysics_judger.aux_judge(data_item['completion'],
    #                                                  data_item['answers'],
    #                                                  data_item['problem'],
    #                                                  data_item['solution'])
    # else:
    #     msg = None

    # score = 1.0 if correctness else -1.0
    return {
        "score": float(correctness),
        "acc": correctness,
        "extracted_gt": extracted_gt,
        "extracted_pred": extracted_pred,
    }
