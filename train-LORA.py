import os, random, logging, sys, argparse, time

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_from_disk
import torch

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}")

################################## ADDED FOR LoRA #####################################
os.system("pip install evaluate")
os.system("pip install rouge_score")
os.system("pip install loralib")
os.system("pip install peft")

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL" # to get more precise error messages

import evaluate
from peft import LoraConfig, get_peft_model, TaskType

def sortout_trainable(model):
    for name, param in model.named_parameters():        
        if 'lora' in name:
            print("name", name)
            param.requires_grad_(True)
        else:
            # REQUIRES to run otherwise ERROR: "Model includes parameters that are not included in the calculation of a loss"
            # BUT by doing that I might block from training other parameters that LORA wants to train and have no "lora" in the name
            # Also, this might cancel/block some additional LORA specific behaviour that we are not aware of
            # Investigate further before commiting to production.
            
            param.requires_grad_(False)
####################################### END #########################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=1) #InExample: 3
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500) # inExample:500
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()

    print("args.output_data_dir", args.output_data_dir)
    print("args.model_dir", args.model_dir)
    print("args.n_gpus", args.n_gpus)
    print("args.training_dir", args.training_dir)
    print("args.test_dir", args.test_dir)

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(level=logging.getLevelName("INFO"),
                        handlers=[logging.StreamHandler(sys.stdout)],
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",)

    # load datasets
    train_dataset = load_from_disk(args.training_dir)
    test_dataset = load_from_disk(args.test_dir)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")

    # compute metrics function for binary classification
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    # download model from model hub
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    #print("ORIGINAL model", model)
    
    print("ORIGINAL model")
    print_trainable_parameters(model)

    ################################## ADDED FOR LoRA #####################################
    from peft import PeftModelForSequenceClassification, get_peft_config
    
    # ADD LoRA adapter/parameters to the Original LLM to be trained
    lora_config = LoraConfig(r=32, # Rank
                             lora_alpha=32,
                             target_modules=["q_lin", "v_lin"],
                             lora_dropout=0.05,
                             bias="all", # InExample: "none"
                             task_type=TaskType.SEQ_CLS)
    
    # Wrap the base model with get_peft_model() to get a trainable PeftModel.
    peft_model = get_peft_model(model, lora_config)
    
    print("PEFT model BEFORE SORTOUT TRINABLES")
    print_trainable_parameters(peft_model)
    
    # Without it there is an ERROR: "Model includes parameters that are not included in the calculation of a loss"
    # Error appears only in the DATA DISTIRBUTED mode; DMP not tested; single GPU no error and this step not needed
    sortout_trainable(peft_model)
    
    print("PEFT model AFTER SORTOUT TRINABLES")
    print_trainable_parameters(peft_model)
    ####################################### END #########################################
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # define training args
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="epoch",
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
    )

    # create Trainer instance
    trainer = Trainer(model=peft_model,
                      args=training_args,
                      compute_metrics=compute_metrics,
                      train_dataset=train_dataset,
                      eval_dataset=test_dataset,
                      tokenizer=tokenizer,)

    # train model
    start_time = time.time()
    trainer.train()
    print("EXEC TIME:", time.time() - start_time)

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=test_dataset)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

    # Saves the model to s3
    trainer.save_model(args.model_dir)
