import argparse
import os
import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from eval.mmlu_pro.categories import subcategories, categories
from eval.utils import get_next_word_predictions, load_hf_tokenizer, load_hf_lm, query_openai_chat_model, dynamic_import_function, upload_results_to_hf, check_and_upload_model_metadata, generate_completions
import vllm
from eval.MATH.examplars import EXAMPLARS as MATH_EXAMPLARS
from eval.MATH.utilities import last_boxed_only_string, remove_boxed
from eval.MATH.minerva_utils import normalize_final_answer, get_unnormalized_answer, is_equiv, r1_normalize_final_answer, mmlu_normalize_final_answer

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

choices = ["A", "B", "C", "D"]
import re
from datasets import load_dataset

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True, cot=False):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1, cot=False):
    if not cot:
        prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
            format_subject(subject)
        )
    else:
        prompt = ''
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def get_demenstrations(k, subject):
    #prompt = ''
    if k == 0:
        return []
    demonstrations = []
    demonstrations.append({
    'question': "A large gene has dozens of exons, of which the central ones code for folded triple helical repeats that connect the cytoskeleton with sarcolemma and extracellular space. Each exon usually codes for one folded triple alpha helix. The most common mutations of the gene are central exon deletions that create out-of-frame peptides and progressive degenerative organ waste. A solution is to deliver a Morpholino that recognizes the 5' end of the out-of-frame exon in pre-mRNA. The molecule prevents binding of the spliceosome and creates exon skipping and in-frame joining. Several missing exons are well tolerated by an organism. Which structure below is not involved in the proposed therapy?",
    'options': 'A. R-loops\nB. lariat\nC. polyA tail\nD. antisense\n',
    'answer': 'Let\'s reason step by step. The text describes the dystrophin gene and the FDA-approved oligonucleotide therapy that causes exon skipping by creating a functional, albeit shorter, dystrophin protein. Morpholino is bound to the pre-mRNA in an antisense orientation. Every splicing mechanism creates the lariat molecule that is circular with a 3\' tail and soon degraded. The spliced RNA is polyadenylated at the 3\' end. R-loops are triple helix of DNA and the pre-mRNA and a consequence of the RNA transcription, not splicing and RNA maturation. Therefore, the correct answer is \\boxed{A}.'
    })
    demonstrations.append({
    'question': "Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they be clearly resolved?",
    'options': 'A. 10^-9 ev\nB. 10^-8 ev\nC. 10^-4 ev\nD. 10^-11 ev\n',
    'answer': 'Let\'s reason step by step. According to the uncertainty principle, Delta E* Delta t=hbar/2. Delta t is the lifetime and Delta E is the width of the energy level. With Delta t=10^-9 s==> Delta E1= 3.3 10^-7 ev. And Delta t=10^-11 s gives Delta E2=3.310^-8 eV. Therefore, the energy difference between the two states must be significantly greater than 10^-7 ev. So the answer is 10^-4 ev. Therefore, the correct answer is \\boxed{C}.'
    })
    demonstrations.append({
    'question': "trans-cinnamaldehyde was treated with methylmagnesium bromide, forming product 1.  1 was treated with pyridinium chlorochromate, forming product 2.  3 was treated with (dimethyl(oxo)-l6-sulfaneylidene)methane in DMSO at elevated temperature, forming product 3.  how many carbon atoms are there in product 3?",
    'options': 'A. 10\nB. 11\nC. 12\nD. 14\n',
    'answer': 'Let\'s reason step by step. trans-cinnamaldehyde was treated with methylmagnesium bromide, forming (E)-4-phenylbut-3-en-2-ol. (E)-4-phenylbut-3-en-2-ol was treated with pyridinium chlorochromate, forming (E)-4-phenylbut-3-en-2-one.  (E)-4-phenylbut-3-en-2-one was treated with (dimethyl(oxo)-l6-sulfaneylidene)methane in DMSO at elevated temperature, forming 1-(2-phenylcyclopropyl)ethan-1-one  Chemical Formula: C11H12O.  Therefore, the correct answer is \\boxed{B}.'
    })

    return demonstrations

@torch.no_grad()
def eval_hf_model(args, subject, model, tokenizer, dev_df, test_df, batch_size=1):
    
    prompts, questions = [], []
    
    demonstrations = get_demenstrations(args.ntrain, subject)
        
    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None
    
    for i in range(0, len(test_df)):
        prompt_end = format_example(test_df, i, include_answer=False, cot=args.cot)
        question = prompt_end
        questions.append(question)
        if args.use_chat_format:
            if args.cot:
                messages = [{"role": "system", "content": f"Please reason step by step, and put your final answer (A, B, C, or D) within \\boxed{{}}."}]
                for demonstration in demonstrations:
                    messages.append({"role": "user", "content": "Question: {}\n {} Reason step by step and put the answer (A, B, C, or D) within \\boxed{{}}.\n".format(demonstration['question'], demonstration['options'])})
                    messages.append({"role": "assistant", "content": "{}\n".format(demonstration['answer'])})
                messages.append({"role": "user", "content": "Question: {}\n Reason step by step and put the answer (A, B, C, or D) in \\boxed{{}}.\n".format(question)})
            else:
                messages = []
                messages.append({"role": "user", "content": "Question: {}".format(question)})
            prompt = chat_formatting_function(messages, tokenizer, add_bos=False)
        else:
            train_prompt = ''
            for demonstration in demonstrations:
                formatted_str = (
                    f"Question: {demonstration['question']}\n"
                    f"{demonstration['options']} "
                    f"Reason step by step and put the answer (A, B, C, or D) in \\boxed{{}}.\n"
                    f"Answer: {demonstration['answer']}\n"
                )
                train_prompt += formatted_str
            prompt = "{} Question: {}\n Reason step by step and put the answer (A, B, C, or D) in \\boxed{{}}.\n Answer: ".format(train_prompt, question)

        tokenized_prompt = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
        if i==1:
            print(prompt)
        prompts.append(prompt)
    # get the answer for all examples
    # adding a prefix space here, as that's expected from the prompt
    # TODO: should raise a warning if this returns more than one token
    
    cors = []
    ground_truths = test_df.iloc[:, -1].values
    stop_string = ['Question:']
    
    if args.use_vllm:
        sampling_params = vllm.SamplingParams(
            temperature=0,
            max_tokens=4096,
            stop=stop_string
        )
        generations = model.generate(prompts, sampling_params)
        prompt_to_output = {
            g.prompt: g.outputs[0].text for g in generations
        }
        outputs = [prompt_to_output[prompt] if prompt in prompt_to_output else "" for prompt in prompts] 
    else:
        outputs = generate_completions(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=1024,
            batch_size=args.eval_batch_size,
            do_sample=False,
        )
            
    predictions = []
        
    for i, output in enumerate(outputs):
        parsed = parse(
            output,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=False,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        if len(parsed) == 0:
            prediction = ''
        elif len(parsed) == 1:
            prediction = parsed[0]
        elif len(parsed) > 1:
            prediction = parsed[-1]
        predictions.append(mmlu_normalize_final_answer(prediction))

    print(len(ground_truths))
    print(len(predictions))
    assert len(predictions) == len(ground_truths)
        
    for i in range(len(predictions)):
        ground_truth = ground_truths[i]
        offset = ord(ground_truth) - ord('A')  # 计算字母偏移量
        correct_column_index = 1 + offset  # 选项列从第2列（索引1）开始
        correct_content = test_df.iloc[i, correct_column_index]  # 获取正确选项内容
        cors.append(predictions[i] == ground_truth or predictions[i] == '{}. {}'.format(ground_truth, correct_content))
        
    acc = np.mean(cors)
    cors = np.array(cors)

    results = [{
        "question": q,
        "answer": ground_truth,
        "model_output": output,
        "prediction": prediction
    } for q, ground_truth, output, prediction in zip(questions, ground_truths, outputs, predictions)]
    
    with open(os.path.join(args.save_dir, "{}.jsonl".format(subject)), "w") as fout:
        for result in results:
            fout.write(json.dumps(result) + "\n")
    all_probs = np.array([[0.25, 0.25, 0.25, 0.25] for _ in range(len(test_df))]) # dummy probs, just don't want to dig into the openai probs
    
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return cors, acc, all_probs


def eval_openai_chat_engine(args, subject, engine, dev_df, test_df, batch_size=1):
    
    import tiktoken
    gpt_tokenizer = tiktoken.get_encoding("cl100k_base")
    answer_choice_ids = [gpt_tokenizer.encode(" " + x)[0] for x in choices]  # be careful, the tokenizer will tokenize " A" and "A" differently.

    prompts = []
    for i in range(0, test_df.shape[0]):
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end        
        prompts.append(prompt)

    instances = [{"id": prompt, "prompt": prompt} for _, prompt in enumerate(prompts)]
    results = query_openai_chat_model(
        engine=args.openai_engine,
        instances=instances,
        batch_size=args.eval_batch_size if args.eval_batch_size else 10,
        output_path=os.path.join(args.save_dir, f"{subject}_openai_results.jsonl"),
        logit_bias={token_id: 100 for token_id in answer_choice_ids},
        max_tokens=1,
    )
    
    # get the metrics
    cors = []
    
    groud_truths = test_df.iloc[:, -1].values
    for i in range(len(test_df)):
        prediction = results[i]["output"].strip()
        ground_truth = groud_truths[i]
        offset = ord(ground_truth) - ord('A')  # 计算字母偏移量
        correct_column_index = 1 + offset  # 选项列从第2列（索引1）开始
        correct_content = test_df.iloc[i, correct_column_index]  # 获取正确选项内容
        cors.append(prediction == ground_truth or prediction == '{}. {}'.format(ground_truth, correct_content))
        
    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array([[0.25, 0.25, 0.25, 0.25] for _ in range(len(test_df))]) # dummy probs, just don't want to dig into the openai probs

    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return cors, acc, all_probs

def main(args):
    print("Loading model and tokenizer...")
    tokenizer = load_hf_tokenizer(
            model_name_or_path=args.model_name_or_path,
            revision=args.hf_revision,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            use_fast_tokenizer=not args.use_slow_tokenizer,
    )
    if not args.use_vllm:
        model = load_hf_lm(
            model_name_or_path=args.model_name_or_path, 
            revision=args.hf_revision,
            load_in_8bit=args.load_in_8bit, 
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            gptq_model=args.gptq,
        )
        from transformers import GPTNeoXForCausalLM, OPTForCausalLM
        if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
            tokenizer.model_max_length = model.config.max_position_embeddings
            print("Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(model.config.max_position_embeddings))
    else:
        model = vllm.LLM(
            model=args.model_name_or_path,
            tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
            tokenizer_mode="slow" if args.use_slow_tokenizer else "auto",
            tensor_parallel_size=torch.cuda.device_count(),
            tokenizer_revision=args.hf_revision,
            revision=args.hf_revision,
        )

        
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if args.subjects:
        assert all(subj in subjects for subj in args.subjects), f"Some of the subjects you specified are not valid: {args.subjects}"
        subjects = args.subjects

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in tqdm(subjects, desc=f"Evaluating subjects: "):
        
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )
        if args.n_instances and args.n_instances < test_df.shape[0]:
            test_df = test_df.sample(args.n_instances, random_state=42)

        if args.model_name_or_path:
            cors, acc, probs = eval_hf_model(args, subject, model, tokenizer, dev_df, test_df, args.eval_batch_size)
        else:
            cors, acc, probs = eval_openai_chat_engine(args, subject, args.openai_engine, dev_df, test_df, args.eval_batch_size)
            
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["correct"] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["choice{}_probs".format(choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                args.save_dir, "{}.csv".format(subject)
            ),
            index=None,
        )

    

    
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))

    # save results
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "average_acc": weighted_acc,
                "cat_acc": {
                    cat: np.mean(cat_cors[cat])
                    for cat in cat_cors
                },
            },
            f,
        )

    if args.upload_to_hf is not None:
        # upload metrics to HF. Main metric is the accuracy
        results = {
            "average_acc": weighted_acc,
            "cat_acc": {
                cat: np.mean(cat_cors[cat])
                for cat in cat_cors
            },
        }
        task_name = f"oi_mmlu_{args.ntrain}shots"
        primary_score = results["average_acc"]
        upload_results_to_hf(
            results,
            args.upload_to_hf,
            args.hf_upload_name,
            task_name=task_name,
            primary_score=primary_score,
            prepend_timestamp=True,
        )
        check_and_upload_model_metadata(
            args.model_name_or_path, args.upload_to_hf, args.hf_upload_name, hf_revision=args.hf_revision
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ntrain",
        type=int,
        default=5
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/mmlu"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/mmlu/llama-7B/"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--hf_revision",
        type=str,
        default=None,
        help="if specified, we will load the model from a revision of the model in the hub"
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--openai_engine",
        type=str,
        default=None,
        help="if specified, we will use the OpenAI API to generate the predictions."
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        help="which subjects to evaluate. If not specified, all the 57 subjects will be evaluated."
    )
    parser.add_argument(
        "--n_instances",
        type=int,
        help="if specified, a maximum of n_instances per subject will be used for the evaluation."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="If given, we're evaluating a 4-bit quantized GPTQ model."
    )
    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument(
        "--upload_to_hf",
        type=str,
        default=None,
        help="If specified, we will upload the results to Hugging Face Datasets. "
             "This should be the name of the dataset to upload to."
    )
    parser.add_argument(
        "--hf_upload_name",
        type=str,
        default=None,
        help="If uploading to hf, this is the model name"
    )
    parser.add_argument(
        "--cot", 
        action="store_true", 
        help="If given, we will use the cot chat format for the prompts."
    )
    parser.add_argument(
        "--use_vllm", 
        action="store_true", 
        help="If given, we will use vllm."
    )
    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    main(args)
