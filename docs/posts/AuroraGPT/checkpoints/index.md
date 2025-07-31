# 💾 Converting Checkpoints
Sam Foreman
2024-10-17

<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://iosevka-webfonts.github.io/iosevka/iosevka.css" rel="stylesheet">

- [MDS –\> HF](#mds--hf)
- [🚧 HF to Meg-DS](#construction-hf-to-meg-ds)
  - [2024-10-17](#2024-10-17)
  - [Older](#older)

## MDS –\> HF

``` bash
convert_mds_to_hf() {
 # GLOBAL_STEP=$1
 CKPT_ROOT=$2

 CKPT_ROOT="/flare/Aurora_deployment/AuroraGPT-Testing/foremans/rollback-41k8/Megatron-DeepSpeed-41800/checkpoints/ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb3072_sp1_pp1_tp1_bf16_optadamw_lr0.00020_lwf0.05/";
 SRC="${CKPT_ROOT}/global_step${GLOBAL_STEP}"
 if [[ -d "${SRC}" ]]; then
        echo "Converting checkpoint @ global step ${GLOBAL_STEP}"
        echo "\tsrc=${SRC}"
        DST="/flare/Aurora_deployment/AuroraGPT-Checkpoints/Megatron-DeepSpeed/checkpoints-to-convert/ws768_ds_stage1_nl32_hs4096_mb4_seq4096_gb3072_sp1_pp1_tp1_bf16_optadamw_lr0.00020_lwf0.05/global_step${GLOBAL_STEP}_hf"
        echo "\tdst=${DST}"
        python3 mds_to_hf.py --mds_checkpoint "${SRC}/mp_rank_00_model_states.pt" --output_dir "${DST}" --cache_dir "./.cache"
 else
        echo "Unable to locate directory ${SRC}. Exiting"
        exit 1
 fi
}
```

# 🚧 HF to Meg-DS

## 2024-10-17

``` python
import os
import ezpz as ez
import torch
import deepspeed
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.integrations import HfDeepSpeedConfig
# distributed setup

os.environ['WORLD_SIZE'] = '12'
rank = ez.setup_torch(backend='deepspeed')
deepspeed.init_distributed()
model_name = "meta-llama/Llama-3.2-1B"
config = AutoConfig.from_pretrained(model_name)
ds_config = {
    "steps_per_print": 1,
    "train_batch_size": 1,
    "train_micro_batch_size_per_gpu": 1,
    "bf16": {
        "enabled": True
    },
    "optimizer": {
        "type": "Adam",
    },
    "zero_optimization": {
        "stage": 3,
    },
}

dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive
# now a model can be loaded.
model = AutoModelForCausalLM.from_pretrained(model_name).to(ez.get_torch_device()).to(torch.bfloat16)
# initialise Deepspeed ZeRO and store only the engine object
ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

``` python
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

dataset = load_dataset("yelp_review_full")
tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

``` python
from transformers import Trainer
training_args = TrainingArguments(output_dir="llama-3.2-1B", deepspeed=ds_config)
trainer = Trainer(model, training_args, train_dataset=small_train_dataset, eval_dataset=small_eval_dataset, tokenizer=tokenizer)
```

## Older

Current status:

``` bash
#[🐍 aurora_nre_models_frameworks-2024.2.1_u1][👻 aurora_nre_models_frameworks-2024.2.1_u1]
#[🌌][03:39:06 PM][foremans]@[x4407c6s7b0n0][/f/A/f/p/a/Megatron-DeepSpeed][🌱hzheng-data-fix][📝🤷🏎💨]
$ _conversion_args=("--hf-ckpt-num-shards 1"
 "--hf-ckpt-dir /flare/Aurora_deployment/foremans/Meta-Llama-3.1-8B-128512"
 "--load-mode auto"
 "--save ckpt-mds-llama-3/"
 "--tensor-model-parallel-size 1"
 "--pipeline-model-parallel-size 1"
 "--lr-warmup-iters 2000"
 "--weight-decay 0.1"
 "--clip-grad 1"
 "--num-layers 32"
 "--hidden-size 4096"
 "--num-attention-heads 32"
 "--ffn-hidden-size 14336"
 "--attention-dropout 0"
 "--hidden-dropout 0"
 "--no-query-key-layer-scaling"
 "--num-key-value-heads 8"
 "--disable-bias-linear"
 "--normalization rmsnorm"
 "--use-rotary-position-embeddings"
 "--untie-embeddings-and-output-weights"
 "--swiglu"
 "--seq-length 2048"
 "--max-position-embeddings 2048"
 "--micro-batch-size 1"
 "--global-batch-size 24"
 "--train-iters 3500"
 "--lr 2e-5"
 "--tensorboard-dir tensorboard_output"
 "--lr-decay-iters 320000"
 "--lr-decay-style cosine"
 "--log-interval 1"
 "--eval-iters 100"
 "--eval-interval 100"
 "--data-path /lus/flare/projects/candle_aesp_CNDA/azton/data/v2/megatron/dataset_v2_wtextbooks_text_document"
 "--save-interval 1500"
 "--split 100,0,0"
 "--bf16"
 "--tokenizer-type HFTokenizer"
 "--tokenizer-model ALCF/custom_tokenizer.model"
 "--deepspeed_config ./examples_deepspeed/finetune_hf_llama/ds_config.json"
 "--deepspeed"
 "--distributed-backend ccl"
 "--no-masked-softmax-fusion"
 "--no-bias-gelu-fusion"
 "--no-bias-dropout-fusion"
 "--no-gradient-accumulation-fusion"
 "--repeated-dataloader"
 "--data-cache-path ./.cache"
 "--make-vocab-size-divisible-by 128512"
 "--vocab-size 128512"
)

conversion_flags=($(printf '%s\n' "${_conversion_args[@]}" | sort))
echo "${conversion_flags[@]}"
--attention-dropout 0 --bf16 --clip-grad 1 --data-cache-path ./.cache --data-path /lus/flare/projects/candle_aesp_CNDA/azton/data/v2/megatron/dataset_v2_wtextbooks_text_document --deepspeed --deepspeed_config ./examples_deepspeed/finetune_hf_llama/ds_config.json --disable-bias-linear --distributed-backend ccl --eval-interval 100 --eval-iters 100 --ffn-hidden-size 14336 --global-batch-size 24 --hf-ckpt-dir /flare/Aurora_deployment/foremans/Meta-Llama-3.1-8B-128512 --hf-ckpt-num-shards 1 --hidden-dropout 0 --hidden-size 4096 --load-mode auto --log-interval 1 --lr 2e-5 --lr-decay-iters 320000 --lr-decay-style cosine --lr-warmup-iters 2000 --make-vocab-size-divisible-by 128512 --max-position-embeddings 2048 --micro-batch-size 1 --no-bias-dropout-fusion --no-bias-gelu-fusion --no-gradient-accumulation-fusion --no-masked-softmax-fusion --no-query-key-layer-scaling --normalization rmsnorm --num-attention-heads 32 --num-key-value-heads 8 --num-layers 32 --pipeline-model-parallel-size 1 --repeated-dataloader --save ckpt-mds-llama-3/ --save-interval 1500 --seq-length 2048 --split 100,0,0 --swiglu --tensorboard-dir tensorboard_output --tensor-model-parallel-size 1 --tokenizer-model ALCF/custom_tokenizer.model --tokenizer-type HFTokenizer --train-iters 3500 --untie-embeddings-and-output-weights --use-rotary-position-embeddings --vocab-size 128512 --weight-decay 0.1
```

``` bash
#[🐍 aurora_nre_models_frameworks-2024.2.1_u1][👻 aurora_nre_models_frameworks-2024.2.1_u1]
#[🌌][03:39:18 PM][foremans]@[x4407c6s7b0n0][/f/A/f/p/a/Megatron-DeepSpeed][🌱hzheng-data-fix][📝🤷🏎💨]
$ launch python3 tools/hf2megads_weight_converter.py "${conversion_flags[@]}"
```

<details closed>

<summary>

output:
</summary>

``` bash
Disabling local launch: multi-node application
Connected to tcp://x4407c6s7b0n0.hostmgmt2407.cm.aurora.alcf.anl.gov:7919
Found executable /flare/Aurora_deployment/foremans/projects/argonne-lcf/Megatron-DeepSpeed/venvs/aurora_nre_models_frameworks-2024.2.1_u1/bin/python3
Launching application bdbd987f-b27b-4922-928b-5d1a166e800b
[2024-10-16 15:39:30,424] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,456] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,456] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,456] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,457] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,568] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,571] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,575] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,578] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,584] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,608] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,613] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,613] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,614] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,615] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,630] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,686] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,698] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,711] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,715] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,716] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,723] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,724] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,726] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,731] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,737] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,755] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,766] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,768] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,780] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,798] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,840] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,870] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,870] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,872] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,873] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,877] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,888] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,896] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,902] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,932] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,934] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,957] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:30,984] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:31,028] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:31,029] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:31,062] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:31,150] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2024-10-16 15:39:33,079] [INFO] [comm.py:161:init_deepspeed_backend] Initialize ccl backend
[2024-10-16 15:39:33,079] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-10-16 15:39:33,079] [INFO] [comm.py:667:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2024-10-16 15:39:33,481] [INFO] [comm.py:161:init_deepspeed_backend] Initialize ccl backend
[2024-10-16 15:39:33,481] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-10-16 15:39:33,481] [INFO] [comm.py:667:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2024-10-16 15:39:33,982] [INFO] [comm.py:161:init_deepspeed_backend] Initialize ccl backend
[2024-10-16 15:39:33,982] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-10-16 15:39:33,982] [INFO] [comm.py:667:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2024-10-16 15:39:33,983] [INFO] [comm.py:161:init_deepspeed_backend] Initialize ccl backend
[2024-10-16 15:39:33,983] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-10-16 15:39:33,983] [INFO] [comm.py:667:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2024-10-16 15:39:33,984] [INFO] [comm.py:161:init_deepspeed_backend] Initialize ccl backend
[2024-10-16 15:39:33,984] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-10-16 15:39:33,984] [INFO] [comm.py:667:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2024-10-16 15:39:34,133] [INFO] [comm.py:161:init_deepspeed_backend] Initialize ccl backend
[2024-10-16 15:39:34,133] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-10-16 15:39:34,133] [INFO] [comm.py:667:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2024-10-16 15:39:34,165] [INFO] [comm.py:161:init_deepspeed_backend] Initialize ccl backend
[2024-10-16 15:39:34,165] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-10-16 15:39:34,165] [INFO] [comm.py:667:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2024-10-16 15:39:34,207] [INFO] [comm.py:161:init_deepspeed_backend] Initialize ccl backend
[2024-10-16 15:39:34,207] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-10-16 15:39:34,207] [INFO] [comm.py:667:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2024-10-16 15:39:34,210] [INFO] [comm.py:161:init_deepspeed_backend] Initialize ccl backend
[2024-10-16 15:39:34,210] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-10-16 15:39:34,210] [INFO] [comm.py:667:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2024-10-16 15:39:34,216] [INFO] [comm.py:161:init_deepspeed_backend] Initialize ccl backend
[2024-10-16 15:39:34,216] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-10-16 15:39:34,216] [INFO] [comm.py:667:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2024-10-16 15:39:34,218] [INFO] [comm.py:161:init_deepspeed_backend] Initialize ccl backend
[2024-10-16 15:39:34,218] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-10-16 15:39:34,219] [INFO] [comm.py:667:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2024-10-16 15:39:34,271] [INFO] [comm.py:161:init_deepspeed_backend] Initialize ccl backend
[2024-10-16 15:39:34,271] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-10-16 15:39:34,271] [INFO] [comm.py:667:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2024-10-16 15:39:34,768] [INFO] [comm.py:161:init_deepspeed_backend] Initialize ccl backend
[2024-10-16 15:39:34,768] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-10-16 15:39:34,768] [INFO] [comm.py:667:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2024-10-16 15:39:34,772] [INFO] [comm.py:161:init_deepspeed_backend] Initialize ccl backend
[2024-10-16 15:39:34,772] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-10-16 15:39:34,772] [INFO] [comm.py:667:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2024-10-16 15:39:34,784] [INFO] [comm.py:161:init_deepspeed_backend] Initialize ccl backend
[2024-10-16 15:39:34,784] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-10-16 15:39:34,784] [INFO] [comm.py:667:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2024-10-16 15:39:34,795] [INFO] [comm.py:161:init_deepspeed_backend] Initialize ccl backend
[2024-10-16 15:39:34,795] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-10-16 15:39:34,795] [INFO] [comm.py:667:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2024-10-16 15:39:35,038] [INFO] [comm.py:161:init_deepspeed_backend] Initialize ccl backend
[2024-10-16 15:39:35,038] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-10-16 15:39:35,038] [INFO] [comm.py:667:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2024-10-16 15:39:35,048] [INFO] [comm.py:161:init_deepspeed_backend] Initialize ccl backend
[2024-10-16 15:39:35,048] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-10-16 15:39:35,048] [INFO] [comm.py:667:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2024-10-16 15:39:35,062] [INFO] [comm.py:161:init_deepspeed_backend] Initialize ccl backend
[2024-10-16 15:39:35,062] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-10-16 15:39:35,062] [INFO] [comm.py:667:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2024-10-16 15:39:35,070] [INFO] [comm.py:161:init_deepspeed_backend] Initialize ccl backend
[2024-10-16 15:39:35,070] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-10-16 15:39:35,070] [INFO] [comm.py:667:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2024-10-16 15:39:35,073] [INFO] [comm.py:161:init_deepspeed_backend] Initialize ccl backend
[2024-10-16 15:39:35,073] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-10-16 15:39:35,073] [INFO] [comm.py:667:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2024-10-16 15:39:35,077] [INFO] [comm.py:161:init_deepspeed_backend] Initialize ccl backend
[2024-10-16 15:39:35,078] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-10-16 15:39:35,078] [INFO] [comm.py:667:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2024-10-16 15:39:35,103] [INFO] [comm.py:161:init_deepspeed_backend] Initialize ccl backend
[2024-10-16 15:39:35,104] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-10-16 15:39:35,104] [INFO] [comm.py:667:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2024-10-16 15:39:35,106] [INFO] [comm.py:161:init_deepspeed_backend] Initialize ccl backend
[2024-10-16 15:39:35,107] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-10-16 15:39:35,107] [INFO] [comm.py:667:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2024-10-16 15:39:35,107] [INFO] [comm.py:718:mpi_discovery] Discovered MPI settings of world_rank=13, local_rank=1, world_size=24, master_addr=10.115.45.184, master_port=29500
[2024-10-16 15:39:35,107] [INFO] [comm.py:718:mpi_discovery] Discovered MPI settings of world_rank=15, local_rank=3, world_size=24, master_addr=10.115.45.184, master_port=29500
[2024-10-16 15:39:35,107] [INFO] [comm.py:718:mpi_discovery] Discovered MPI settings of world_rank=16, local_rank=4, world_size=24, master_addr=10.115.45.184, master_port=29500
[2024-10-16 15:39:35,107] [INFO] [comm.py:718:mpi_discovery] Discovered MPI settings of world_rank=17, local_rank=5, world_size=24, master_addr=10.115.45.184, master_port=29500
[2024-10-16 15:39:35,107] [INFO] [comm.py:718:mpi_discovery] Discovered MPI settings of world_rank=12, local_rank=0, world_size=24, master_addr=10.115.45.184, master_port=29500
[2024-10-16 15:39:35,107] [INFO] [comm.py:718:mpi_discovery] Discovered MPI settings of world_rank=14, local_rank=2, world_size=24, master_addr=10.115.45.184, master_port=29500
[2024-10-16 15:39:35,107] [INFO] [comm.py:718:mpi_discovery] Discovered MPI settings of world_rank=0, local_rank=0, world_size=24, master_addr=10.115.45.184, master_port=29500
[2024-10-16 15:39:35,107] [INFO] [comm.py:683:init_distributed] Initializing TorchBackend in DeepSpeed with backend ccl
[2024-10-16 15:39:35,107] [INFO] [comm.py:718:mpi_discovery] Discovered MPI settings of world_rank=18, local_rank=6, world_size=24, master_addr=10.115.45.184, master_port=29500
[2024-10-16 15:39:35,107] [INFO] [comm.py:718:mpi_discovery] Discovered MPI settings of world_rank=19, local_rank=7, world_size=24, master_addr=10.115.45.184, master_port=29500
[2024-10-16 15:39:35,107] [INFO] [comm.py:718:mpi_discovery] Discovered MPI settings of world_rank=20, local_rank=8, world_size=24, master_addr=10.115.45.184, master_port=29500
[2024-10-16 15:39:35,107] [INFO] [comm.py:718:mpi_discovery] Discovered MPI settings of world_rank=21, local_rank=9, world_size=24, master_addr=10.115.45.184, master_port=29500
[2024-10-16 15:39:35,107] [INFO] [comm.py:718:mpi_discovery] Discovered MPI settings of world_rank=22, local_rank=10, world_size=24, master_addr=10.115.45.184, master_port=29500
[2024-10-16 15:39:35,107] [INFO] [comm.py:718:mpi_discovery] Discovered MPI settings of world_rank=23, local_rank=11, world_size=24, master_addr=10.115.45.184, master_port=29500
[2024-10-16 15:39:35,107] [INFO] [comm.py:718:mpi_discovery] Discovered MPI settings of world_rank=1, local_rank=1, world_size=24, master_addr=10.115.45.184, master_port=29500
[2024-10-16 15:39:35,107] [INFO] [comm.py:718:mpi_discovery] Discovered MPI settings of world_rank=2, local_rank=2, world_size=24, master_addr=10.115.45.184, master_port=29500
[2024-10-16 15:39:35,107] [INFO] [comm.py:718:mpi_discovery] Discovered MPI settings of world_rank=3, local_rank=3, world_size=24, master_addr=10.115.45.184, master_port=29500
[2024-10-16 15:39:35,107] [INFO] [comm.py:718:mpi_discovery] Discovered MPI settings of world_rank=4, local_rank=4, world_size=24, master_addr=10.115.45.184, master_port=29500
[2024-10-16 15:39:35,107] [INFO] [comm.py:718:mpi_discovery] Discovered MPI settings of world_rank=5, local_rank=5, world_size=24, master_addr=10.115.45.184, master_port=29500
[2024-10-16 15:39:35,107] [INFO] [comm.py:718:mpi_discovery] Discovered MPI settings of world_rank=6, local_rank=6, world_size=24, master_addr=10.115.45.184, master_port=29500
[2024-10-16 15:39:35,107] [INFO] [comm.py:718:mpi_discovery] Discovered MPI settings of world_rank=7, local_rank=7, world_size=24, master_addr=10.115.45.184, master_port=29500
[2024-10-16 15:39:35,107] [INFO] [comm.py:718:mpi_discovery] Discovered MPI settings of world_rank=8, local_rank=8, world_size=24, master_addr=10.115.45.184, master_port=29500
[2024-10-16 15:39:35,107] [INFO] [comm.py:718:mpi_discovery] Discovered MPI settings of world_rank=9, local_rank=9, world_size=24, master_addr=10.115.45.184, master_port=29500
[2024-10-16 15:39:35,107] [INFO] [comm.py:718:mpi_discovery] Discovered MPI settings of world_rank=10, local_rank=10, world_size=24, master_addr=10.115.45.184, master_port=29500
[2024-10-16 15:39:35,107] [INFO] [comm.py:718:mpi_discovery] Discovered MPI settings of world_rank=11, local_rank=11, world_size=24, master_addr=10.115.45.184, master_port=29500
[2024-10-16 15:39:35.116490][INFO][dist.py:348] - [device='xpu'][rank=15/23][local_rank=3/11][node=1/1]
[2024-10-16 15:39:35.117455][INFO][dist.py:348] - [device='xpu'][rank=1/23][local_rank=1/11][node=1/1]
2024:10:16-15:39:35:(34466) |CCL_WARN| MPI was initialized externally, CCL-MPI specific environment is ignored
2024:10:16-15:39:35:(169066) |CCL_WARN| MPI was initialized externally, CCL-MPI specific environment is ignored
[2024-10-16 15:39:35.120281][INFO][dist.py:348] - [device='xpu'][rank=10/23][local_rank=10/11][node=0/1]
[2024-10-16 15:39:35.120280][INFO][dist.py:348] - [device='xpu'][rank=11/23][local_rank=11/11][node=1/1]
2024:10:16-15:39:35:(169075) |CCL_WARN| MPI was initialized externally, CCL-MPI specific environment is ignored
2024:10:16-15:39:35:(169076) |CCL_WARN| MPI was initialized externally, CCL-MPI specific environment is ignored
[2024-10-16 15:39:35.124730][INFO][dist.py:348] - [device='xpu'][rank=4/23][local_rank=4/11][node=0/1]
[2024-10-16 15:39:35.125260][INFO][dist.py:348] - [device='xpu'][rank=6/23][local_rank=6/11][node=0/1]
[2024-10-16 15:39:35.125427][INFO][dist.py:348] - [device='xpu'][rank=3/23][local_rank=3/11][node=1/1]
2024:10:16-15:39:35:(169069) |CCL_WARN| MPI was initialized externally, CCL-MPI specific environment is ignored
2024:10:16-15:39:35:(169068) |CCL_WARN| MPI was initialized externally, CCL-MPI specific environment is ignored
2024:10:16-15:39:35:(169071) |CCL_WARN| MPI was initialized externally, CCL-MPI specific environment is ignored
[2024-10-16 15:39:35.127167][INFO][dist.py:348] - [device='xpu'][rank=8/23][local_rank=8/11][node=0/1]
[2024-10-16 15:39:35.127199][INFO][dist.py:348] - [device='xpu'][rank=21/23][local_rank=9/11][node=1/1]
2024:10:16-15:39:35:(169073) |CCL_WARN| MPI was initialized externally, CCL-MPI specific environment is ignored
2024:10:16-15:39:35:(34472) |CCL_WARN| MPI was initialized externally, CCL-MPI specific environment is ignored
[2024-10-16 15:39:35.129097][INFO][dist.py:348] - [device='xpu'][rank=14/23][local_rank=2/11][node=0/1]
[2024-10-16 15:39:35.129281][INFO][dist.py:348] - [device='xpu'][rank=9/23][local_rank=9/11][node=1/1]
[2024-10-16 15:39:35.129345][INFO][dist.py:348] - [device='xpu'][rank=22/23][local_rank=10/11][node=0/1]
[2024-10-16 15:39:35.129461][INFO][dist.py:348] - [device='xpu'][rank=20/23][local_rank=8/11][node=0/1]
2024:10:16-15:39:35:(34465) |CCL_WARN| MPI was initialized externally, CCL-MPI specific environment is ignored
[2024-10-16 15:39:35.130518][INFO][dist.py:348] - [device='xpu'][rank=13/23][local_rank=1/11][node=1/1]
2024:10:16-15:39:35:(34473) |CCL_WARN| MPI was initialized externally, CCL-MPI specific environment is ignored
2024:10:16-15:39:35:(169074) |CCL_WARN| MPI was initialized externally, CCL-MPI specific environment is ignored
2024:10:16-15:39:35:(34471) |CCL_WARN| MPI was initialized externally, CCL-MPI specific environment is ignored
2024:10:16-15:39:35:(34464) |CCL_WARN| MPI was initialized externally, CCL-MPI specific environment is ignored
[2024-10-16 15:39:35.131822][INFO][dist.py:348] - [device='xpu'][rank=18/23][local_rank=6/11][node=0/1]
[2024-10-16 15:39:35.131816][INFO][dist.py:348] - [device='xpu'][rank=19/23][local_rank=7/11][node=1/1]
2024:10:16-15:39:35:(34469) |CCL_WARN| MPI was initialized externally, CCL-MPI specific environment is ignored
[2024-10-16 15:39:35.133183][INFO][dist.py:348] - [device='xpu'][rank=7/23][local_rank=7/11][node=1/1]
2024:10:16-15:39:35:(34470) |CCL_WARN| MPI was initialized externally, CCL-MPI specific environment is ignored
2024:10:16-15:39:35:(169072) |CCL_WARN| MPI was initialized externally, CCL-MPI specific environment is ignored
[2024-10-16 15:39:35.192746][INFO][dist.py:348] - [device='xpu'][rank=2/23][local_rank=2/11][node=0/1]
[2024-10-16 15:39:35.192682][INFO][dist.py:348] - [device='xpu'][rank=5/23][local_rank=5/11][node=1/1]
[2024-10-16 15:39:35.193446][INFO][dist.py:348] - [device='xpu'][rank=17/23][local_rank=5/11][node=1/1]
2024:10:16-15:39:35:(169067) |CCL_WARN| MPI was initialized externally, CCL-MPI specific environment is ignored
[2024-10-16 15:39:35.193506][INFO][dist.py:348] - [device='xpu'][rank=12/23][local_rank=0/11][node=0/1]
2024:10:16-15:39:35:(169070) |CCL_WARN| MPI was initialized externally, CCL-MPI specific environment is ignored
[2024-10-16 15:39:35.194435][INFO][dist.py:348] - [device='xpu'][rank=23/23][local_rank=11/11][node=1/1]
[2024-10-16 15:39:35.194529][INFO][dist.py:348] - [device='xpu'][rank=16/23][local_rank=4/11][node=0/1]
2024:10:16-15:39:35:(34468) |CCL_WARN| MPI was initialized externally, CCL-MPI specific environment is ignored
2024:10:16-15:39:35:(34463) |CCL_WARN| MPI was initialized externally, CCL-MPI specific environment is ignored
2024:10:16-15:39:35:(34467) |CCL_WARN| MPI was initialized externally, CCL-MPI specific environment is ignored
2024:10:16-15:39:35:(34474) |CCL_WARN| MPI was initialized externally, CCL-MPI specific environment is ignored
[2024-10-16 15:39:35.198383][INFO][dist.py:92] -

[dist_info]:
  • DEVICE=xpu
  • DEVICE_ID=xpu:0
  • DISTRIBUTED_BACKEND=ccl
  • GPUS_PER_NODE=12
  • HOSTS=['x4407c6s7b0n0.hostmgmt2407.cm.aurora.alcf.anl.gov', 'x4308c2s5b0n0.hostmgmt2308.cm.aurora.alcf.anl.gov']
  • HOSTFILE=/var/spool/pbs/aux/886439.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
  • HOSTNAME=x4407c6s7b0n0.hostmgmt2407.cm.aurora.alcf.anl.gov
  • LOCAL_RANK=0
  • MACHINE=Aurora
  • NUM_NODES=2
  • NGPUS=24
  • NGPUS_AVAILABLE=24
  • NODE_ID=0
  • RANK=0
  • SCHEDULER=PBS
  • WORLD_SIZE_TOTAL=24
  • WORLD_SIZE_IN_USE=24
  • LAUNCH_CMD=mpiexec --verbose --envall -n 24 -ppn 12 --hostfile /var/spool/pbs/aux/886439.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --cpu-bind depth -d 16


--------------------------------------------------
DeepSpeed C++/CUDA extension op report
--------------------------------------------------
NOTE: Ops not installed will be just-in-time (JIT) compiled at
      runtime if needed. Op compatibility means that your system
      meet the required dependencies to JIT install the op.
--------------------------------------------------
JIT compiled ops requires ninja
ninja .................. [OKAY]
--------------------------------------------------
op name ................ installed .. compatible
--------------------------------------------------
deepspeed_not_implemented  [NO] ....... [OKAY]
 [WARNING]  async_io requires the dev libaio .so object and headers but these were not found.
 [WARNING]  If libaio is already installed (perhaps from source), try setting the CFLAGS and LDFLAGS environment variables to where it can be found.
async_io ............... [NO] ....... [NO]
cpu_adagrad ............ [NO] ....... [OKAY]
cpu_adam ............... [NO] ....... [OKAY]
flash_attn ............. [NO] ....... [OKAY]
fused_adam ............. [NO] ....... [OKAY]
transformer_inference .. [NO] ....... [OKAY]
pack_bits .............. [NO] ....... [OKAY]
--------------------------------------------------
DeepSpeed general environment info:
torch install path ............... ['/opt/aurora/24.180.0/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages/torch']
torch version .................... 2.3.1+cxx11.abi
deepspeed install path ........... ['/lus/flare/projects/Aurora_deployment/foremans/projects/argonne-lcf/Megatron-DeepSpeed/deps/DeepSpeed/deepspeed']
deepspeed info ................... 0.15.3+unknown, unknown, unknown
deepspeed wheel compiled w. ...... torch 2.3
shared memory (/dev/shm) size .... 503.18 GB
[2024-10-16 15:39:35.319255][INFO][configs.py:272] - **** Git info for DeepSpeed: git_hash=7ef26bf git_branch=hzheng-data-fix ****
[2024-10-16 15:39:35.319927][INFO][dist.py:725] - Using oneccl_bindings from: /opt/aurora/24.180.0/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages/oneccl_bindings_for_pytorch/__init__.py
[2024-10-16 15:39:35.320347][INFO][dist.py:727] - Using ipex from: /opt/aurora/24.180.0/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages/intel_extension_for_pytorch/__init__.py
[2024-10-16 15:39:35.320734][INFO][dist.py:728] - [0/24] Using device='xpu' with backend='deepspeed' + 'ccl' for distributed training.
[2024-10-16 15:39:35.325748][INFO][dist.py:348] - [device='xpu'][rank=0/23][local_rank=0/11][node=0/1]
[2024-10-16 15:39:35.326291][WARNING][_logger.py:68] - Using [24 / 24] available "xpu" devices !!
2024:10:16-15:39:35:(169065) |CCL_WARN| value of CCL_KVS_MODE changed to be mpi (default:pmi)
2024:10:16-15:39:35:(169065) |CCL_WARN| value of CCL_KVS_CONNECTION_TIMEOUT changed to be 3600 (default:120)
2024:10:16-15:39:35:(169065) |CCL_WARN| value of CCL_BCAST changed to be double_tree (default:)
2024:10:16-15:39:35:(169065) |CCL_WARN| value of CCL_ENABLE_SYCL_KERNELS changed to be 1 (default:0)
2024:10:16-15:39:35:(169065) |CCL_WARN| value of CCL_SYCL_ESIMD changed to be 1 (default:0)
2024:10:16-15:39:35:(169065) |CCL_WARN| value of CCL_PROCESS_LAUNCHER changed to be pmix (default:hydra)
2024:10:16-15:39:35:(169065) |CCL_WARN| value of CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD changed to be 32768 (default:1000)
2024:10:16-15:39:35:(169065) |CCL_WARN| CCL_ALLGATHERV_MEDIUM_SIZE_THRESHOLD=0 is unknown to and unused by oneCCL code but is present in the environment, check if it is not mistyped.
2024:10:16-15:39:35:(169065) |CCL_WARN| CCL_SKIP_SCHEDULER=1 is unknown to and unused by oneCCL code but is present in the environment, check if it is not mistyped.
2024:10:16-15:39:35:(169065) |CCL_WARN| MPI was initialized externally, CCL-MPI specific environment is ignored
using world size: 24, data-parallel-size: 24, sequence-parallel size: 1, tensor-model-parallel size: 1, pipeline-model-parallel size: 1
accumulate and all-reduce gradients in fp32 for bfloat16 data type.
using torch.bfloat16 for parameters ...
------------------------ arguments ------------------------
  accumulate_allreduce_grads_in_fp32 .............. True
  adam_beta1 ...................................... 0.9
  adam_beta2 ...................................... 0.999
  adam_eps ........................................ 1e-08
  add_bias_linear ................................. False
  add_position_embedding .......................... False
  adlr_autoresume ................................. False
  adlr_autoresume_interval ........................ 1000
  aml_data_download_path .......................... None
  apply_layernorm_1p .............................. False
  apply_query_key_layer_scaling ................... False
  apply_residual_connection_post_layernorm ........ False
  async_tensor_model_parallel_allreduce ........... False
  attention_dropout ............................... 0.0
  attention_softmax_in_fp32 ....................... False
  barrier_with_L1_time ............................ True
  bert_binary_head ................................ True
  bert_embedder_type .............................. megatron
  bert_load ....................................... None
  bf16 ............................................ True
  bias_dropout_fusion ............................. False
  bias_gelu_fusion ................................ False
  biencoder_projection_dim ........................ 0
  biencoder_shared_query_context_model ............ False
  block_data_path ................................. None
  checkpoint_activations .......................... False
  checkpoint_in_cpu ............................... False
  checkpoint_num_layers ........................... 1
  classes_fraction ................................ 1.0
  clip_grad ....................................... 1.0
  compression_training ............................ False
  consumed_train_samples .......................... 0
  consumed_train_tokens ........................... 0
  consumed_valid_samples .......................... 0
  contigious_checkpointing ........................ False
  cpu_optimizer ................................... False
  cpu_torch_adam .................................. False
  create_moe_param_group .......................... False
  curriculum_learning_legacy ...................... False
  data_cache_path ................................. ./.cache
  data_efficiency_curriculum_learning ............. False
  data_file_list .................................. None
  data_impl ....................................... infer
  data_parallel_random_init ....................... False
  data_parallel_size .............................. 24
  data_path ....................................... ['/lus/flare/projects/candle_aesp_CNDA/azton/data/v2/megatron/dataset_v2_wtextbooks_text_document']
  data_per_class_fraction ......................... 1.0
  data_sharding ................................... True
  dataloader_type ................................. single
  DDP_impl ........................................ local
  decoder_num_layers .............................. None
  decoder_seq_length .............................. None
  deepscale ....................................... False
  deepscale_config ................................ None
  deepspeed ....................................... True
  deepspeed_activation_checkpointing .............. False
  deepspeed_config ................................ ./examples_deepspeed/finetune_hf_llama/ds_config.json
  dino_bottleneck_size ............................ 256
  dino_freeze_last_layer .......................... 1
  dino_head_hidden_size ........................... 2048
  dino_local_crops_number ......................... 10
  dino_local_img_size ............................. 96
  dino_norm_last_layer ............................ False
  dino_teacher_temp ............................... 0.07
  dino_warmup_teacher_temp ........................ 0.04
  dino_warmup_teacher_temp_epochs ................. 30
  distribute_checkpointed_activations ............. False
  distribute_saved_activations .................... False
  distributed_backend ............................. ccl
  distributed_timeout_minutes ..................... 10
  ds_fused_adam ................................... False
  ds_inference .................................... False
  ds_pipeline_enabled ............................. True
  ds_sequence_parallel_size ....................... 1
  embedding_path .................................. None
  embedding_weights_in_fp32 ....................... False
  empty_unused_memory_level ....................... 0
  enable_expert_tensor_parallelism ................ False
  enable_zbh1_exact_semantics ..................... False
  enable_zbh1_pipeline ............................ False
  encoder_num_layers .............................. 32
  encoder_seq_length .............................. 2048
  end_weight_decay ................................ 0.1
  eod_mask_loss ................................... False
  eval_interval ................................... 100
  eval_iters ...................................... 100
  evidence_data_path .............................. None
  exit_duration_in_mins ........................... None
  exit_interval ................................... None
  exit_on_missing_checkpoint ...................... False
  exit_signal_handler ............................. False
  expert_interval ................................. 2
  ffn_hidden_size ................................. 14336
  finetune ........................................ False
  force_ds_sequence_parallel ...................... False
  fp16 ............................................ False
  fp16_lm_cross_entropy ........................... False
  fp32_residual_connection ........................ False
  fp8_amax_compute_algo ........................... most_recent
  fp8_amax_history_len ............................ 1
  fp8_e4m3 ........................................ False
  fp8_hybrid ...................................... False
  fp8_interval .................................... 1
  fp8_margin ...................................... 0
  fp8_wgrad ....................................... True
  global_batch_size ............................... 24
  gradient_accumulation_fusion .................... False
  head_lr_mult .................................... 1.0
  hf_ckpt_dir ..................................... /flare/Aurora_deployment/foremans/Meta-Llama-3.1-8B-128512
  hf_ckpt_num_shards .............................. 1
  hidden_dropout .................................. 0.0
  hidden_size ..................................... 4096
  hidden_size_teacher ............................. None
  hysteresis ...................................... 2
  ict_head_size ................................... None
  ict_load ........................................ None
  img_h ........................................... 224
  img_w ........................................... 224
  indexer_batch_size .............................. 128
  indexer_log_interval ............................ 1000
  inference ....................................... False
  inference_batch_times_seqlen_threshold .......... 512
  init_method_std ................................. 0.02
  init_method_xavier_uniform ...................... False
  initial_loss_scale .............................. 4294967296
  iter_per_epoch .................................. 1250
  kd .............................................. False
  kd_alpha_ce ..................................... 1
  kd_beta_ce ...................................... 1
  kd_temp ......................................... 1.0
  kill_switch_file ................................ None
  kv_channels ..................................... 128
  layernorm_epsilon ............................... 1e-05
  lazy_mpu_init ................................... None
  load ............................................ None
  load_mode ....................................... auto
  load_tag ........................................ None
  load_teacher .................................... None
  local_rank ...................................... None
  log_batch_size_to_tensorboard ................... False
  log_interval .................................... 1
  log_learning_rate_to_tensorboard ................ True
  log_loss_scale_to_tensorboard ................... True
  log_memory_to_tensorboard ....................... False
  log_num_zeros_in_grad ........................... False
  log_optimizer_states_to_tensorboard ............. False
  log_params_norm ................................. False
  log_timers_to_tensorboard ....................... False
  log_validation_ppl_to_tensorboard ............... False
  log_world_size_to_tensorboard ................... False
  loss_scale ...................................... None
  loss_scale_window ............................... 1000
  lr .............................................. 2e-05
  lr_decay_iters .................................. 320000
  lr_decay_samples ................................ None
  lr_decay_style .................................. cosine
  lr_decay_tokens ................................. None
  lr_warmup_fraction .............................. None
  lr_warmup_iters ................................. 2000
  lr_warmup_samples ............................... 0
  lr_warmup_tokens ................................ None
  make_vocab_size_divisible_by .................... 128512
  mask_factor ..................................... 1.0
  mask_prob ....................................... 0.15
  mask_type ....................................... random
  masked_softmax_fusion ........................... False
  max_position_embeddings ......................... 2048
  max_tokens_to_oom ............................... 12000
  mem_efficient_ln ................................ True
  memory_centric_tiled_linear ..................... False
  merge_file ...................................... None
  micro_batch_size ................................ 1
  min_loss_scale .................................. 1.0
  min_lr .......................................... 0.0
  mlp_type ........................................ standard
  mmap_warmup ..................................... False
  moe_eval_capacity_factor ........................ 1.0
  moe_expert_parallel_size ........................ 1
  moe_loss_coeff .................................. 0.1
  moe_min_capacity ................................ 4
  moe_token_dropping .............................. True
  moe_top2_2nd_expert_sampling .................... True
  moe_train_capacity_factor ....................... 1.0
  mos ............................................. False
  multiprocessing_context ......................... fork
  no_load_lr_state ................................ False
  no_load_optim ................................... None
  no_load_rng ..................................... None
  no_persist_layer_norm ........................... False
  no_pipeline_parallel ............................ False
  no_save_optim ................................... None
  no_save_rng ..................................... None
  normalization ................................... rmsnorm
  num_attention_heads ............................. 32
  num_attention_heads_teacher ..................... None
  num_channels .................................... 3
  num_classes ..................................... 1000
  num_experts ..................................... [1]
  num_experts_switch .............................. None
  num_experts_teacher ............................. [1]
  num_key_value_heads ............................. 8
  num_layers ...................................... 32
  num_layers_per_virtual_pipeline_stage ........... None
  num_layers_teacher .............................. None
  num_workers ..................................... 2
  onnx_safe ....................................... None
  openai_gelu ..................................... False
  optimizer ....................................... adam
  output_bert_embeddings .......................... False
  overlap_p2p_comm ................................ False
  override_opt_param_scheduler .................... False
  params_dtype .................................... torch.bfloat16
  partition_activations ........................... False
  patch_dim ....................................... 16
  perform_initialization .......................... True
  pipeline_model_parallel_size .................... 1
  pipeline_model_parallel_split_rank .............. None
  profile ......................................... None
  profile_backward ................................ False
  profile_ranks ................................... None
  profile_steps ................................... 2,3
  query_in_block_prob ............................. 0.1
  rampup_batch_size ............................... None
  random_ltd ...................................... False
  rank ............................................ 0
  recompute_granularity ........................... None
  recompute_method ................................ None
  recompute_num_layers ............................ 1
  remote_device ................................... none
  repeated_dataloader ............................. True
  reset_attention_mask ............................ False
  reset_iteration ................................. False
  reset_position_ids .............................. False
  retriever_report_topk_accuracies ................ []
  retriever_score_scaling ......................... False
  retriever_seq_length ............................ 256
  retro_add_retriever ............................. False
  retro_cyclic_train_iters ........................ None
  retro_encoder_attention_dropout ................. 0.1
  retro_encoder_hidden_dropout .................... 0.1
  retro_encoder_layers ............................ 2
  retro_num_neighbors ............................. 2
  retro_num_retrieved_chunks ...................... 2
  retro_return_doc_ids ............................ False
  retro_workdir ................................... None
  return_data_index ............................... False
  rope_theta ...................................... 10000
  rotary_percent .................................. 1.0
  sample_rate ..................................... 1.0
  save ............................................ ckpt-mds-llama-3/
  save_interval ................................... 1500
  scatter_gather_tensors_in_pipeline .............. True
  scattered_embeddings ............................ False
  schedulefree_for_each ........................... False
  seed ............................................ 1234
  seq_length ...................................... 2048
  sequence_parallel ............................... False
  sgd_momentum .................................... 0.9
  short_seq_prob .................................. 0.1
  shuffle_sample .................................. False
  skip_train ...................................... False
  sophiag_beta1 ................................... 0.9
  sophiag_beta2 ................................... 0.95
  sophiag_rho ..................................... 0.01
  split ........................................... 100,0,0
  split_transformers .............................. False
  squared_relu .................................... False
  standalone_embedding_stage ...................... False
  start_weight_decay .............................. 0.1
  swiglu .......................................... True
  swin_backbone_type .............................. tiny
  synchronize_each_layer .......................... False
  tensor_model_parallel_size ...................... 1
  tensorboard_dir ................................. tensorboard_output
  tensorboard_log_interval ........................ 1
  tensorboard_queue_size .......................... 1000
  test_data_path .................................. None
  tile_factor ..................................... 1
  timing_log_level ................................ 0
  timing_log_option ............................... minmax
  titles_data_path ................................ None
  to_hf_ckpt ...................................... False
  tokenizer_model ................................. ALCF/custom_tokenizer.model
  tokenizer_type .................................. HFTokenizer
  topk ............................................ 1
  trace_dir ....................................... ./trace/
  train_data_exact_num_epochs ..................... None
  train_data_path ................................. None
  train_desc_path ................................. None
  train_doc_idx_path .............................. None
  train_idx_path .................................. None
  train_iters ..................................... 3500
  train_iters_to_skip ............................. None
  train_range_to_skip ............................. None
  train_sample_idx_path ........................... None
  train_samples ................................... None
  train_shuffle_idx_path .......................... None
  train_tokens .................................... None
  transformer_impl ................................ local
  transformer_pipeline_model_parallel_size ........ 1
  trust_remote_code ............................... False
  universal_checkpoint ............................ False
  untie_embeddings_and_output_weights ............. True
  use_checkpoint_args ............................. False
  use_checkpoint_opt_param_scheduler .............. False
  use_contiguous_buffers_in_local_ddp ............. True
  use_cpu_initialization .......................... None
  use_dataset_only ................................ False
  use_distributed_optimizer ....................... False
  use_flash_attn .................................. False
  use_flash_attn_builder .......................... False
  use_flash_attn_triton ........................... False
  use_flash_attn_v1 ............................... False
  use_flash_attn_v2 ............................... False
  use_mics ........................................ False
  use_one_sent_docs ............................... False
  use_pin_memory .................................. False
  use_ring_exchange_p2p ........................... False
  use_rotary_position_embeddings .................. True
  use_tutel ....................................... False
  valid_data_path ................................. None
  variable_seq_lengths ............................ False
  virtual_pipeline_model_parallel_size ............ None
  vision_backbone_type ............................ vit
  vision_pretraining .............................. False
  vision_pretraining_type ......................... classify
  vocab_extra_ids ................................. 0
  vocab_file ...................................... None
  vocab_size ...................................... 128512
  wandb_exp_name ..................................
  wandb_project ...................................
  wandb_save_dir ..................................
  weight_decay .................................... 0.1
  weight_decay_incr_style ......................... constant
  world_size ...................................... 24
  zero_allgather_bucket_size ...................... 0.0
  zero_contigious_gradients ....................... False
  zero_reduce_bucket_size ......................... 0.0
  zero_reduce_scatter ............................. False
  zero_stage ...................................... 1.0
-------------------- end of arguments ---------------------
setting number of micro-batches to constant 1
> building HFTokenizer tokenizer ...
 > padded vocab (size: 128000) with 512 dummy tokens (new size: 128512)
torch distributed is already initialized, skipping initialization ...
> initialized tensor model parallel with size 1
> initialized pipeline model parallel with size 1
> setting random seeds to 1234 ...
> initializing model parallel cuda seeds on global rank 0, model parallel rank 0, and data parallel rank 0 with model parallel seed: 3952 and data parallel seed: 1234
>fused kernel is only supported in cuda, skip loading fused kernel
[2024-10-16 15:39:36,876] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
>fused kernel is only supported in cuda, skip loading fused kernel
[2024-10-16 15:39:36,876] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
>fused kernel is only supported in cuda, skip loading fused kernel
[2024-10-16 15:39:36,876] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
>fused kernel is only supported in cuda, skip loading fused kernel
[2024-10-16 15:39:36,876] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
>fused kernel is only supported in cuda, skip loading fused kernel
[2024-10-16 15:39:36,879] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
>fused kernel is only supported in cuda, skip loading fused kernel
[2024-10-16 15:39:36,881] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
>fused kernel is only supported in cuda, skip loading fused kernel
[2024-10-16 15:39:36,881] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
>fused kernel is only supported in cuda, skip loading fused kernel
[2024-10-16 15:39:36,882] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
>fused kernel is only supported in cuda, skip loading fused kernel
[2024-10-16 15:39:36,882] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
make: Entering directory '/lus/flare/projects/Aurora_deployment/foremans/projects/argonne-lcf/Megatron-DeepSpeed/megatron/data'
>fused kernel is only supported in cuda, skip loading fused kernel
[2024-10-16 15:39:36,884] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
>fused kernel is only supported in cuda, skip loading fused kernel
[2024-10-16 15:39:36,922] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
>fused kernel is only supported in cuda, skip loading fused kernel
[2024-10-16 15:39:36,929] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
>fused kernel is only supported in cuda, skip loading fused kernel
[2024-10-16 15:39:36,930] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
>fused kernel is only supported in cuda, skip loading fused kernel
[2024-10-16 15:39:36,931] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
>fused kernel is only supported in cuda, skip loading fused kernel
[2024-10-16 15:39:36,935] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
>fused kernel is only supported in cuda, skip loading fused kernel
[2024-10-16 15:39:36,937] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
>fused kernel is only supported in cuda, skip loading fused kernel
[2024-10-16 15:39:36,939] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
>fused kernel is only supported in cuda, skip loading fused kernel
[2024-10-16 15:39:36,939] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
>fused kernel is only supported in cuda, skip loading fused kernel
[2024-10-16 15:39:36,940] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
>fused kernel is only supported in cuda, skip loading fused kernel
[2024-10-16 15:39:36,941] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
>fused kernel is only supported in cuda, skip loading fused kernel
[2024-10-16 15:39:36,946] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
>fused kernel is only supported in cuda, skip loading fused kernel
[2024-10-16 15:39:36,949] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
make: Nothing to be done for 'default'.
make: Leaving directory '/lus/flare/projects/Aurora_deployment/foremans/projects/argonne-lcf/Megatron-DeepSpeed/megatron/data'
> compiling dataset index builder ...
>>> done with dataset index builder. Compilation time: 0.080 seconds
>fused kernel is only supported in cuda, skip loading fused kernel
[2024-10-16 15:39:36.956560][INFO][hf2megads_weight_converter.py:479] - building model ...
[2024-10-16 15:39:37,133] [INFO] [utils.py:781:see_memory_usage] Before Building Model
[2024-10-16 15:39:37,133] [INFO] [utils.py:782:see_memory_usage] MA 0.0 GB         Max_MA 0.0 GB         CA 0.0 GB         Max_CA 0 GB
[2024-10-16 15:39:37,133] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 105.24 GB, percent = 9.3%
[2024-10-16 15:39:37,133] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
SEED_LAYERS=False BASE_SEED=1234 SEED_FN=None
Using topology: {ProcessCoord(pipe=0, data=0, model=0): 0, ProcessCoord(pipe=0, data=1, model=0): 1, ProcessCoord(pipe=0, data=2, model=0): 2, ProcessCoord(pipe=0, data=3, model=0): 3, ProcessCoord(pipe=0, data=4, model=0): 4, ProcessCoord(pipe=0, data=5, model=0): 5, ProcessCoord(pipe=0, data=6, model=0): 6, ProcessCoord(pipe=0, data=7, model=0): 7, ProcessCoord(pipe=0, data=8, model=0): 8, ProcessCoord(pipe=0, data=9, model=0): 9, ProcessCoord(pipe=0, data=10, model=0): 10, ProcessCoord(pipe=0, data=11, model=0): 11, ProcessCoord(pipe=0, data=12, model=0): 12, ProcessCoord(pipe=0, data=13, model=0): 13, ProcessCoord(pipe=0, data=14, model=0): 14, ProcessCoord(pipe=0, data=15, model=0): 15, ProcessCoord(pipe=0, data=16, model=0): 16, ProcessCoord(pipe=0, data=17, model=0): 17, ProcessCoord(pipe=0, data=18, model=0): 18, ProcessCoord(pipe=0, data=19, model=0): 19, ProcessCoord(pipe=0, data=20, model=0): 20, ProcessCoord(pipe=0, data=21, model=0): 21, ProcessCoord(pipe=0, data=22, model=0): 22, ProcessCoord(pipe=0, data=23, model=0): 23}
[2024-10-16 15:39:37,139] [INFO] [module.py:396:_partition_layers] Partitioning pipeline stages with method type:transformer
2024-10-16 15:39:37.212904: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-10-16 15:39:37.212925: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-10-16 15:39:37.213970: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2024-10-16 15:39:37.704522: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Loading checkpoint shards:   0%|          | 0/7 [00:00<?, ?it/s]1 GPTModelPipe(
  (tied_modules): ModuleDict()
  (1): EmbeddingPipe(
    (word_embeddings): VocabParallelEmbedding()
    (embedding_dropout): Dropout(p=0.0, inplace=False)
  )
  (2): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (3): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (4): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (5): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (6): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (7): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (8): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (9): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (10): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (11): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (12): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (13): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (14): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (15): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (16): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (17): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (18): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (19): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (20): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (21): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (22): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (23): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (24): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (25): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (26): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (27): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (28): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (29): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (30): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (31): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (32): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (33): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (34): RMSNorm()
  (35): LMHeadPipe(
    (lm_head): ColumnParallelLinear()
  )
Loading checkpoint shards:   0%|          | 0/7 [00:00<?, ?it/s]
Loading checkpoint shards:   0%|          | 0/7 [00:00<?, ?it/s]stage=0 layers=37
     0: _to_float16
     1: EmbeddingPipe
     2: ParallelTransformerLayerPipe
     3: ParallelTransformerLayerPipe
     4: ParallelTransformerLayerPipe
     5: ParallelTransformerLayerPipe
     6: ParallelTransformerLayerPipe
     7: ParallelTransformerLayerPipe
     8: ParallelTransformerLayerPipe
     9: ParallelTransformerLayerPipe
    10: ParallelTransformerLayerPipe
    11: ParallelTransformerLayerPipe
    12: ParallelTransformerLayerPipe
    13: ParallelTransformerLayerPipe
    14: ParallelTransformerLayerPipe
    15: ParallelTransformerLayerPipe
    16: ParallelTransformerLayerPipe
    17: ParallelTransformerLayerPipe
    18: ParallelTransformerLayerPipe
    19: ParallelTransformerLayerPipe
    20: ParallelTransformerLayerPipe
    21: ParallelTransformerLayerPipe
    22: ParallelTransformerLayerPipe
    23: ParallelTransformerLayerPipe
    24: ParallelTransformerLayerPipe
    25: ParallelTransformerLayerPipe
    26: ParallelTransformerLayerPipe
    27: ParallelTransformerLayerPipe
    28: ParallelTransformerLayerPipe
    29: ParallelTransformerLayerPipe
    30: ParallelTransformerLayerPipe
    31: ParallelTransformerLayerPipe
    32: ParallelTransformerLayerPipe
    33: ParallelTransformerLayerPipe
    34: RMSNorm
    35: LMHeadPipe
    36: float16_to_fp32
  loss: loss_func
[2024-10-16 15:39:38,077] [INFO] [utils.py:781:see_memory_usage] After Building Model
[2024-10-16 15:39:38,077] [INFO] [utils.py:782:see_memory_usage] MA 14.96 GB         Max_MA 14.96 GB         CA 14.96 GB         Max_CA 15 GB
[2024-10-16 15:39:38,077] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 105.4 GB, percent = 9.3%
0 GPTModelPipe(
  (tied_modules): ModuleDict()
  (1): EmbeddingPipe(
    (word_embeddings): VocabParallelEmbedding()
    (embedding_dropout): Dropout(p=0.0, inplace=False)
  )
  (2): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (3): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (4): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (5): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (6): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (7): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (8): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (9): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (10): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (11): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (12): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (13): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (14): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (15): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (16): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (17): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (18): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (19): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (20): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (21): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (22): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (23): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (24): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (25): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (26): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (27): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (28): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (29): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (30): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (31): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (32): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (33): ParallelTransformerLayerPipe(
    (input_layernorm): RMSNorm()
    (self_attention): ParallelAttention(
      (query_key_value): ColumnParallelLinear()
      (core_attention): CoreAttention(
        (scale_mask_softmax): FusedScaleMaskSoftmax()
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (dense): RowParallelLinear()
    )
    (post_attention_layernorm): RMSNorm()
    (mlp): ParallelMLP(
      (dense_h_to_4h): ColumnParallelLinear()
      (dense_4h_to_h): RowParallelLinear()
    )
  )
  (34): RMSNorm()
  (35): LMHeadPipe(
    (lm_head): ColumnParallelLinear()
  )
)
Loading checkpoint shards:   0%|          | 0/7 [00:00<?, ?it/s]2024-10-16 15:39:39.334375: W external/local_tsl/tsl/lib/monitoring/collection_registry.cc:81] Trying to register 2 metrics with the same name: /tensorflow/core/bfc_allocator_delay. The old value will be erased inorder to register a new one. Please check if you link the metric more than once, or if the name is already used by other metrics.
2024-10-16 15:39:39.334561: W external/local_tsl/tsl/lib/monitoring/collection_registry.cc:81] Trying to register 2 metrics with the same name: /xla/service/gpu/compiled_programs_count. The old value will be erased in order to register a new one. Please check if you link the metric more than once, or if the name is already used by other metrics.
2024-10-16 15:39:39.335711: W external/local_tsl/tsl/lib/monitoring/collection_registry.cc:81] Trying to register 2 metrics with the same name: /jax/pjrt/pjrt_executable_executions. The old value will be erased in order to register a new one. Please check if you link the metric more than once, or if the name is already used by other metrics.
2024-10-16 15:39:39.335722: W external/local_tsl/tsl/lib/monitoring/collection_registry.cc:81] Trying to register 2 metrics with the same name: /jax/pjrt/pjrt_executable_execution_time_usecs. The old value will be erased in order to register a new one. Please check if you linkthe metric more than once, or if the name is already used by other metrics.
2024-10-16 15:39:39.587872: I itex/core/wrapper/itex_gpu_wrapper.cc:38] Intel Extension for Tensorflow* GPU backend is loaded.
2024-10-16 15:39:39.631109: I itex/core/devices/gpu/itex_gpu_runtime.cc:130] Selected platform: Intel(R) Level-Zero
2024-10-16 15:39:39.631594: I itex/core/devices/gpu/itex_gpu_runtime.cc:155] number of sub-devices is zero, expose root device.
2024-10-16 15:39:39.631598: I itex/core/devices/gpu/itex_gpu_runtime.cc:155] number of sub-devices is zero, expose root device.
2024-10-16 15:39:39.631600: I itex/core/devices/gpu/itex_gpu_runtime.cc:155] number of sub-devices is zero, expose root device.
2024-10-16 15:39:39.631602: I itex/core/devices/gpu/itex_gpu_runtime.cc:155] number of sub-devices is zero, expose root device.
2024-10-16 15:39:39.631604: I itex/core/devices/gpu/itex_gpu_runtime.cc:155] number of sub-devices is zero, expose root device.
2024-10-16 15:39:39.631607: I itex/core/devices/gpu/itex_gpu_runtime.cc:155] number of sub-devices is zero, expose root device.
2024-10-16 15:39:39.631609: I itex/core/devices/gpu/itex_gpu_runtime.cc:155] number of sub-devices is zero, expose root device.
2024-10-16 15:39:39.631611: I itex/core/devices/gpu/itex_gpu_runtime.cc:155] number of sub-devices is zero, expose root device.
2024-10-16 15:39:39.631613: I itex/core/devices/gpu/itex_gpu_runtime.cc:155] number of sub-devices is zero, expose root device.
2024-10-16 15:39:39.631615: I itex/core/devices/gpu/itex_gpu_runtime.cc:155] number of sub-devices is zero, expose root device.
2024-10-16 15:39:39.631617: I itex/core/devices/gpu/itex_gpu_runtime.cc:155] number of sub-devices is zero, expose root device.
2024-10-16 15:39:39.631619: I itex/core/devices/gpu/itex_gpu_runtime.cc:155] number of sub-devices is zero, expose root device.
> setting tensorboard ...
WARNING: WANDB writing requested but no legit wandb project or experiment name provided, therefore no WANDB logs will be written according to random generated project or experiment name.
>fused kernel is only supported in cuda, skip loading fused kernel
[2024-10-16 15:39:39,835] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
Loading checkpoint shards: 100%|██████████| 7/7 [03:35<00:00, 30.78s/it]
[2024-10-16 15:43:13.654241][INFO][hf2megads_weight_converter.py:108] - ----------------------------hf weight list----------------------------
[2024-10-16 15:43:13.705041][INFO][hf2megads_weight_converter.py:113] - model.embed_tokens.weight
[2024-10-16 15:43:13.708051][INFO][hf2megads_weight_converter.py:113] - model.layers.0.self_attn.q_proj.weight
[2024-10-16 15:43:13.710201][INFO][hf2megads_weight_converter.py:113] - model.layers.0.self_attn.k_proj.weight
[2024-10-16 15:43:13.711855][INFO][hf2megads_weight_converter.py:113] - model.layers.0.self_attn.v_proj.weight
[2024-10-16 15:43:13.714676][INFO][hf2megads_weight_converter.py:113] - model.layers.0.self_attn.o_proj.weight
[2024-10-16 15:43:13.721608][INFO][hf2megads_weight_converter.py:113] - model.layers.0.mlp.gate_proj.weight
[2024-10-16 15:43:13.728527][INFO][hf2megads_weight_converter.py:113] - model.layers.0.mlp.up_proj.weight
[2024-10-16 15:43:13.735447][INFO][hf2megads_weight_converter.py:113] - model.layers.0.mlp.down_proj.weight
[2024-10-16 15:43:13.736224][INFO][hf2megads_weight_converter.py:113] - model.layers.0.input_layernorm.weight
[2024-10-16 15:43:13.736764][INFO][hf2megads_weight_converter.py:113] - model.layers.0.post_attention_layernorm.weight
[2024-10-16 15:43:13.739392][INFO][hf2megads_weight_converter.py:113] - model.layers.1.self_attn.q_proj.weight
[2024-10-16 15:43:13.741013][INFO][hf2megads_weight_converter.py:113] - model.layers.1.self_attn.k_proj.weight
[2024-10-16 15:43:13.742600][INFO][hf2megads_weight_converter.py:113] - model.layers.1.self_attn.v_proj.weight
[2024-10-16 15:43:13.745359][INFO][hf2megads_weight_converter.py:113] - model.layers.1.self_attn.o_proj.weight
[2024-10-16 15:43:13.752286][INFO][hf2megads_weight_converter.py:113] - model.layers.1.mlp.gate_proj.weight
[2024-10-16 15:43:13.759192][INFO][hf2megads_weight_converter.py:113] - model.layers.1.mlp.up_proj.weight
[2024-10-16 15:43:13.766054][INFO][hf2megads_weight_converter.py:113] - model.layers.1.mlp.down_proj.weight
[2024-10-16 15:43:13.766785][INFO][hf2megads_weight_converter.py:113] - model.layers.1.input_layernorm.weight
[2024-10-16 15:43:13.767321][INFO][hf2megads_weight_converter.py:113] - model.layers.1.post_attention_layernorm.weight
[2024-10-16 15:43:13.769938][INFO][hf2megads_weight_converter.py:113] - model.layers.2.self_attn.q_proj.weight
[2024-10-16 15:43:13.771536][INFO][hf2megads_weight_converter.py:113] - model.layers.2.self_attn.k_proj.weight
[2024-10-16 15:43:13.773107][INFO][hf2megads_weight_converter.py:113] - model.layers.2.self_attn.v_proj.weight
[2024-10-16 15:43:13.775861][INFO][hf2megads_weight_converter.py:113] - model.layers.2.self_attn.o_proj.weight
[2024-10-16 15:43:13.782733][INFO][hf2megads_weight_converter.py:113] - model.layers.2.mlp.gate_proj.weight
[2024-10-16 15:43:13.789559][INFO][hf2megads_weight_converter.py:113] - model.layers.2.mlp.up_proj.weight
[2024-10-16 15:43:13.796385][INFO][hf2megads_weight_converter.py:113] - model.layers.2.mlp.down_proj.weight
[2024-10-16 15:43:13.797080][INFO][hf2megads_weight_converter.py:113] - model.layers.2.input_layernorm.weight
[2024-10-16 15:43:13.797626][INFO][hf2megads_weight_converter.py:113] - model.layers.2.post_attention_layernorm.weight
[2024-10-16 15:43:13.800331][INFO][hf2megads_weight_converter.py:113] - model.layers.3.self_attn.q_proj.weight
[2024-10-16 15:43:13.801911][INFO][hf2megads_weight_converter.py:113] - model.layers.3.self_attn.k_proj.weight
[2024-10-16 15:43:13.803453][INFO][hf2megads_weight_converter.py:113] - model.layers.3.self_attn.v_proj.weight
[2024-10-16 15:43:13.806474][INFO][hf2megads_weight_converter.py:113] - model.layers.3.self_attn.o_proj.weight
Loading checkpoint shards: 100%|██████████| 7/7 [03:35<00:00, 30.84s/it]
[2024-10-16 15:43:13.813328][INFO][hf2megads_weight_converter.py:113] - model.layers.3.mlp.gate_proj.weight
[2024-10-16 15:43:13.820151][INFO][hf2megads_weight_converter.py:113] - model.layers.3.mlp.up_proj.weight
[2024-10-16 15:43:13.826971][INFO][hf2megads_weight_converter.py:113] - model.layers.3.mlp.down_proj.weight
[2024-10-16 15:43:13.827640][INFO][hf2megads_weight_converter.py:113] - model.layers.3.input_layernorm.weight
[2024-10-16 15:43:13.828185][INFO][hf2megads_weight_converter.py:113] - model.layers.3.post_attention_layernorm.weight
[2024-10-16 15:43:13.831062][INFO][hf2megads_weight_converter.py:113] - model.layers.4.self_attn.q_proj.weight
[2024-10-16 15:43:13.832590][INFO][hf2megads_weight_converter.py:113] - model.layers.4.self_attn.k_proj.weight
[2024-10-16 15:43:13.834171][INFO][hf2megads_weight_converter.py:113] - model.layers.4.self_attn.v_proj.weight
[2024-10-16 15:43:13.837147][INFO][hf2megads_weight_converter.py:113] - model.layers.4.self_attn.o_proj.weight
[2024-10-16 15:43:13.843970][INFO][hf2megads_weight_converter.py:113] - model.layers.4.mlp.gate_proj.weight
[2024-10-16 15:43:13.850793][INFO][hf2megads_weight_converter.py:113] - model.layers.4.mlp.up_proj.weight
[2024-10-16 15:43:13.857596][INFO][hf2megads_weight_converter.py:113] - model.layers.4.mlp.down_proj.weight
[2024-10-16 15:43:13.858284][INFO][hf2megads_weight_converter.py:113] - model.layers.4.input_layernorm.weight
[2024-10-16 15:43:13.858810][INFO][hf2megads_weight_converter.py:113] - model.layers.4.post_attention_layernorm.weight
[2024-10-16 15:43:13.861661][INFO][hf2megads_weight_converter.py:113] - model.layers.5.self_attn.q_proj.weight
[2024-10-16 15:43:13.863209][INFO][hf2megads_weight_converter.py:113] - model.layers.5.self_attn.k_proj.weight
[2024-10-16 15:43:13.864753][INFO][hf2megads_weight_converter.py:113] - model.layers.5.self_attn.v_proj.weight
[2024-10-16 15:43:13.867739][INFO][hf2megads_weight_converter.py:113] - model.layers.5.self_attn.o_proj.weight
[2024-10-16 15:43:13.874537][INFO][hf2megads_weight_converter.py:113] - model.layers.5.mlp.gate_proj.weight
[2024-10-16 15:43:13.881318][INFO][hf2megads_weight_converter.py:113] - model.layers.5.mlp.up_proj.weight
[2024-10-16 15:43:13.888097][INFO][hf2megads_weight_converter.py:113] - model.layers.5.mlp.down_proj.weight
[2024-10-16 15:43:13.888767][INFO][hf2megads_weight_converter.py:113] - model.layers.5.input_layernorm.weight
[2024-10-16 15:43:13.889295][INFO][hf2megads_weight_converter.py:113] - model.layers.5.post_attention_layernorm.weight
[2024-10-16 15:43:13.892151][INFO][hf2megads_weight_converter.py:113] - model.layers.6.self_attn.q_proj.weight
[2024-10-16 15:43:13.893641][INFO][hf2megads_weight_converter.py:113] - model.layers.6.self_attn.k_proj.weight
[2024-10-16 15:43:13.895211][INFO][hf2megads_weight_converter.py:113] - model.layers.6.self_attn.v_proj.weight
[2024-10-16 15:43:13.898154][INFO][hf2megads_weight_converter.py:113] - model.layers.6.self_attn.o_proj.weight
[2024-10-16 15:43:13.904976][INFO][hf2megads_weight_converter.py:113] - model.layers.6.mlp.gate_proj.weight
[2024-10-16 15:43:13.911767][INFO][hf2megads_weight_converter.py:113] - model.layers.6.mlp.up_proj.weight
[2024-10-16 15:43:13.918536][INFO][hf2megads_weight_converter.py:113] - model.layers.6.mlp.down_proj.weight
[2024-10-16 15:43:13.919201][INFO][hf2megads_weight_converter.py:113] - model.layers.6.input_layernorm.weight
[2024-10-16 15:43:13.919795][INFO][hf2megads_weight_converter.py:113] - model.layers.6.post_attention_layernorm.weight
[2024-10-16 15:43:13.922646][INFO][hf2megads_weight_converter.py:113] - model.layers.7.self_attn.q_proj.weight
[2024-10-16 15:43:13.924200][INFO][hf2megads_weight_converter.py:113] - model.layers.7.self_attn.k_proj.weight
[2024-10-16 15:43:13.925687][INFO][hf2megads_weight_converter.py:113] - model.layers.7.self_attn.v_proj.weight
[2024-10-16 15:43:13.928651][INFO][hf2megads_weight_converter.py:113] - model.layers.7.self_attn.o_proj.weight
[2024-10-16 15:43:13.935432][INFO][hf2megads_weight_converter.py:113] - model.layers.7.mlp.gate_proj.weight
[2024-10-16 15:43:13.942178][INFO][hf2megads_weight_converter.py:113] - model.layers.7.mlp.up_proj.weight
[2024-10-16 15:43:13.948916][INFO][hf2megads_weight_converter.py:113] - model.layers.7.mlp.down_proj.weight
[2024-10-16 15:43:13.949547][INFO][hf2megads_weight_converter.py:113] - model.layers.7.input_layernorm.weight
[2024-10-16 15:43:13.950059][INFO][hf2megads_weight_converter.py:113] - model.layers.7.post_attention_layernorm.weight
[2024-10-16 15:43:13.952902][INFO][hf2megads_weight_converter.py:113] - model.layers.8.self_attn.q_proj.weight
[2024-10-16 15:43:13.954461][INFO][hf2megads_weight_converter.py:113] - model.layers.8.self_attn.k_proj.weight
[2024-10-16 15:43:13.955985][INFO][hf2megads_weight_converter.py:113] - model.layers.8.self_attn.v_proj.weight
[2024-10-16 15:43:13.958931][INFO][hf2megads_weight_converter.py:113] - model.layers.8.self_attn.o_proj.weight
[2024-10-16 15:43:13.965709][INFO][hf2megads_weight_converter.py:113] - model.layers.8.mlp.gate_proj.weight
[2024-10-16 15:43:13.972481][INFO][hf2megads_weight_converter.py:113] - model.layers.8.mlp.up_proj.weight
[2024-10-16 15:43:13.979242][INFO][hf2megads_weight_converter.py:113] - model.layers.8.mlp.down_proj.weight
[2024-10-16 15:43:13.979876][INFO][hf2megads_weight_converter.py:113] - model.layers.8.input_layernorm.weight
[2024-10-16 15:43:13.980381][INFO][hf2megads_weight_converter.py:113] - model.layers.8.post_attention_layernorm.weight
[2024-10-16 15:43:13.983353][INFO][hf2megads_weight_converter.py:113] - model.layers.9.self_attn.q_proj.weight
[2024-10-16 15:43:13.984910][INFO][hf2megads_weight_converter.py:113] - model.layers.9.self_attn.k_proj.weight
[2024-10-16 15:43:13.986401][INFO][hf2megads_weight_converter.py:113] - model.layers.9.self_attn.v_proj.weight
[2024-10-16 15:43:13.989279][INFO][hf2megads_weight_converter.py:113] - model.layers.9.self_attn.o_proj.weight
[2024-10-16 15:43:13.996056][INFO][hf2megads_weight_converter.py:113] - model.layers.9.mlp.gate_proj.weight
[2024-10-16 15:43:14.002856][INFO][hf2megads_weight_converter.py:113] - model.layers.9.mlp.up_proj.weight
[2024-10-16 15:43:14.009601][INFO][hf2megads_weight_converter.py:113] - model.layers.9.mlp.down_proj.weight
[2024-10-16 15:43:14.010234][INFO][hf2megads_weight_converter.py:113] - model.layers.9.input_layernorm.weight
[2024-10-16 15:43:14.010742][INFO][hf2megads_weight_converter.py:113] - model.layers.9.post_attention_layernorm.weight
[2024-10-16 15:43:14.013552][INFO][hf2megads_weight_converter.py:113] - model.layers.10.self_attn.q_proj.weight
[2024-10-16 15:43:14.015117][INFO][hf2megads_weight_converter.py:113] - model.layers.10.self_attn.k_proj.weight
[2024-10-16 15:43:14.016607][INFO][hf2megads_weight_converter.py:113] - model.layers.10.self_attn.v_proj.weight
[2024-10-16 15:43:14.019542][INFO][hf2megads_weight_converter.py:113] - model.layers.10.self_attn.o_proj.weight
[2024-10-16 15:43:14.026297][INFO][hf2megads_weight_converter.py:113] - model.layers.10.mlp.gate_proj.weight
[2024-10-16 15:43:14.033038][INFO][hf2megads_weight_converter.py:113] - model.layers.10.mlp.up_proj.weight
[2024-10-16 15:43:14.039752][INFO][hf2megads_weight_converter.py:113] - model.layers.10.mlp.down_proj.weight
[2024-10-16 15:43:14.040455][INFO][hf2megads_weight_converter.py:113] - model.layers.10.input_layernorm.weight
[2024-10-16 15:43:14.040966][INFO][hf2megads_weight_converter.py:113] - model.layers.10.post_attention_layernorm.weight
[2024-10-16 15:43:14.043760][INFO][hf2megads_weight_converter.py:113] - model.layers.11.self_attn.q_proj.weight
[2024-10-16 15:43:14.045346][INFO][hf2megads_weight_converter.py:113] - model.layers.11.self_attn.k_proj.weight
[2024-10-16 15:43:14.046849][INFO][hf2megads_weight_converter.py:113] - model.layers.11.self_attn.v_proj.weight
[2024-10-16 15:43:14.049725][INFO][hf2megads_weight_converter.py:113] - model.layers.11.self_attn.o_proj.weight
[2024-10-16 15:43:14.056435][INFO][hf2megads_weight_converter.py:113] - model.layers.11.mlp.gate_proj.weight
[2024-10-16 15:43:14.063163][INFO][hf2megads_weight_converter.py:113] - model.layers.11.mlp.up_proj.weight
[2024-10-16 15:43:14.069898][INFO][hf2megads_weight_converter.py:113] - model.layers.11.mlp.down_proj.weight
[2024-10-16 15:43:14.070561][INFO][hf2megads_weight_converter.py:113] - model.layers.11.input_layernorm.weight
[2024-10-16 15:43:14.071084][INFO][hf2megads_weight_converter.py:113] - model.layers.11.post_attention_layernorm.weight
[2024-10-16 15:43:14.073869][INFO][hf2megads_weight_converter.py:113] - model.layers.12.self_attn.q_proj.weight
[2024-10-16 15:43:14.075363][INFO][hf2megads_weight_converter.py:113] - model.layers.12.self_attn.k_proj.weight
[2024-10-16 15:43:14.076918][INFO][hf2megads_weight_converter.py:113] - model.layers.12.self_attn.v_proj.weight
[2024-10-16 15:43:14.079777][INFO][hf2megads_weight_converter.py:113] - model.layers.12.self_attn.o_proj.weight
[2024-10-16 15:43:14.086497][INFO][hf2megads_weight_converter.py:113] - model.layers.12.mlp.gate_proj.weight
[2024-10-16 15:43:14.093183][INFO][hf2megads_weight_converter.py:113] - model.layers.12.mlp.up_proj.weight
[2024-10-16 15:43:14.099869][INFO][hf2megads_weight_converter.py:113] - model.layers.12.mlp.down_proj.weight
[2024-10-16 15:43:14.100523][INFO][hf2megads_weight_converter.py:113] - model.layers.12.input_layernorm.weight
[2024-10-16 15:43:14.101038][INFO][hf2megads_weight_converter.py:113] - model.layers.12.post_attention_layernorm.weight
[2024-10-16 15:43:14.103823][INFO][hf2megads_weight_converter.py:113] - model.layers.13.self_attn.q_proj.weight
[2024-10-16 15:43:14.105335][INFO][hf2megads_weight_converter.py:113] - model.layers.13.self_attn.k_proj.weight
[2024-10-16 15:43:14.106828][INFO][hf2megads_weight_converter.py:113] - model.layers.13.self_attn.v_proj.weight
[2024-10-16 15:43:14.109698][INFO][hf2megads_weight_converter.py:113] - model.layers.13.self_attn.o_proj.weight
[2024-10-16 15:43:14.116395][INFO][hf2megads_weight_converter.py:113] - model.layers.13.mlp.gate_proj.weight
[2024-10-16 15:43:14.123086][INFO][hf2megads_weight_converter.py:113] - model.layers.13.mlp.up_proj.weight
[2024-10-16 15:43:14.129807][INFO][hf2megads_weight_converter.py:113] - model.layers.13.mlp.down_proj.weight
[2024-10-16 15:43:14.130474][INFO][hf2megads_weight_converter.py:113] - model.layers.13.input_layernorm.weight
[2024-10-16 15:43:14.130997][INFO][hf2megads_weight_converter.py:113] - model.layers.13.post_attention_layernorm.weight
[2024-10-16 15:43:14.133762][INFO][hf2megads_weight_converter.py:113] - model.layers.14.self_attn.q_proj.weight
[2024-10-16 15:43:14.135290][INFO][hf2megads_weight_converter.py:113] - model.layers.14.self_attn.k_proj.weight
[2024-10-16 15:43:14.136791][INFO][hf2megads_weight_converter.py:113] - model.layers.14.self_attn.v_proj.weight
[2024-10-16 15:43:14.139860][INFO][hf2megads_weight_converter.py:113] - model.layers.14.self_attn.o_proj.weight
[2024-10-16 15:43:14.146560][INFO][hf2megads_weight_converter.py:113] - model.layers.14.mlp.gate_proj.weight
[2024-10-16 15:43:14.153229][INFO][hf2megads_weight_converter.py:113] - model.layers.14.mlp.up_proj.weight
[2024-10-16 15:43:14.160012][INFO][hf2megads_weight_converter.py:113] - model.layers.14.mlp.down_proj.weight
[2024-10-16 15:43:14.160681][INFO][hf2megads_weight_converter.py:113] - model.layers.14.input_layernorm.weight
[2024-10-16 15:43:14.161212][INFO][hf2megads_weight_converter.py:113] - model.layers.14.post_attention_layernorm.weight
[2024-10-16 15:43:14.164011][INFO][hf2megads_weight_converter.py:113] - model.layers.15.self_attn.q_proj.weight
[2024-10-16 15:43:14.165550][INFO][hf2megads_weight_converter.py:113] - model.layers.15.self_attn.k_proj.weight
[2024-10-16 15:43:14.167029][INFO][hf2megads_weight_converter.py:113] - model.layers.15.self_attn.v_proj.weight
[2024-10-16 15:43:14.169860][INFO][hf2megads_weight_converter.py:113] - model.layers.15.self_attn.o_proj.weight
[2024-10-16 15:43:14.176522][INFO][hf2megads_weight_converter.py:113] - model.layers.15.mlp.gate_proj.weight
[2024-10-16 15:43:14.183206][INFO][hf2megads_weight_converter.py:113] - model.layers.15.mlp.up_proj.weight
[2024-10-16 15:43:14.189866][INFO][hf2megads_weight_converter.py:113] - model.layers.15.mlp.down_proj.weight
[2024-10-16 15:43:14.190530][INFO][hf2megads_weight_converter.py:113] - model.layers.15.input_layernorm.weight
[2024-10-16 15:43:14.191065][INFO][hf2megads_weight_converter.py:113] - model.layers.15.post_attention_layernorm.weight
[2024-10-16 15:43:14.193838][INFO][hf2megads_weight_converter.py:113] - model.layers.16.self_attn.q_proj.weight
[2024-10-16 15:43:14.195331][INFO][hf2megads_weight_converter.py:113] - model.layers.16.self_attn.k_proj.weight
[2024-10-16 15:43:14.196892][INFO][hf2megads_weight_converter.py:113] - model.layers.16.self_attn.v_proj.weight
[2024-10-16 15:43:14.199748][INFO][hf2megads_weight_converter.py:113] - model.layers.16.self_attn.o_proj.weight
[2024-10-16 15:43:14.206446][INFO][hf2megads_weight_converter.py:113] - model.layers.16.mlp.gate_proj.weight
[2024-10-16 15:43:14.213109][INFO][hf2megads_weight_converter.py:113] - model.layers.16.mlp.up_proj.weight
[2024-10-16 15:43:14.219783][INFO][hf2megads_weight_converter.py:113] - model.layers.16.mlp.down_proj.weight
[2024-10-16 15:43:14.220402][INFO][hf2megads_weight_converter.py:113] - model.layers.16.input_layernorm.weight
[2024-10-16 15:43:14.220932][INFO][hf2megads_weight_converter.py:113] - model.layers.16.post_attention_layernorm.weight
[2024-10-16 15:43:14.223676][INFO][hf2megads_weight_converter.py:113] - model.layers.17.self_attn.q_proj.weight
[2024-10-16 15:43:14.225232][INFO][hf2megads_weight_converter.py:113] - model.layers.17.self_attn.k_proj.weight
[2024-10-16 15:43:14.226737][INFO][hf2megads_weight_converter.py:113] - model.layers.17.self_attn.v_proj.weight
[2024-10-16 15:43:14.229543][INFO][hf2megads_weight_converter.py:113] - model.layers.17.self_attn.o_proj.weight
[2024-10-16 15:43:14.236240][INFO][hf2megads_weight_converter.py:113] - model.layers.17.mlp.gate_proj.weight
[2024-10-16 15:43:14.242898][INFO][hf2megads_weight_converter.py:113] - model.layers.17.mlp.up_proj.weight
[2024-10-16 15:43:14.249590][INFO][hf2megads_weight_converter.py:113] - model.layers.17.mlp.down_proj.weight
[2024-10-16 15:43:14.250235][INFO][hf2megads_weight_converter.py:113] - model.layers.17.input_layernorm.weight
[2024-10-16 15:43:14.250747][INFO][hf2megads_weight_converter.py:113] - model.layers.17.post_attention_layernorm.weight
[2024-10-16 15:43:14.253465][INFO][hf2megads_weight_converter.py:113] - model.layers.18.self_attn.q_proj.weight
[2024-10-16 15:43:14.255062][INFO][hf2megads_weight_converter.py:113] - model.layers.18.self_attn.k_proj.weight
[2024-10-16 15:43:14.256546][INFO][hf2megads_weight_converter.py:113] - model.layers.18.self_attn.v_proj.weight
[2024-10-16 15:43:14.259362][INFO][hf2megads_weight_converter.py:113] - model.layers.18.self_attn.o_proj.weight
[2024-10-16 15:43:14.266006][INFO][hf2megads_weight_converter.py:113] - model.layers.18.mlp.gate_proj.weight
[2024-10-16 15:43:14.272677][INFO][hf2megads_weight_converter.py:113] - model.layers.18.mlp.up_proj.weight
[2024-10-16 15:43:14.279406][INFO][hf2megads_weight_converter.py:113] - model.layers.18.mlp.down_proj.weight
[2024-10-16 15:43:14.280055][INFO][hf2megads_weight_converter.py:113] - model.layers.18.input_layernorm.weight
[2024-10-16 15:43:14.280566][INFO][hf2megads_weight_converter.py:113] - model.layers.18.post_attention_layernorm.weight
[2024-10-16 15:43:14.283295][INFO][hf2megads_weight_converter.py:113] - model.layers.19.self_attn.q_proj.weight
[2024-10-16 15:43:14.284803][INFO][hf2megads_weight_converter.py:113] - model.layers.19.self_attn.k_proj.weight
[2024-10-16 15:43:14.286350][INFO][hf2megads_weight_converter.py:113] - model.layers.19.self_attn.v_proj.weight
[2024-10-16 15:43:14.289142][INFO][hf2megads_weight_converter.py:113] - model.layers.19.self_attn.o_proj.weight
[2024-10-16 15:43:14.295818][INFO][hf2megads_weight_converter.py:113] - model.layers.19.mlp.gate_proj.weight
[2024-10-16 15:43:14.302488][INFO][hf2megads_weight_converter.py:113] - model.layers.19.mlp.up_proj.weight
[2024-10-16 15:43:14.309098][INFO][hf2megads_weight_converter.py:113] - model.layers.19.mlp.down_proj.weight
[2024-10-16 15:43:14.309731][INFO][hf2megads_weight_converter.py:113] - model.layers.19.input_layernorm.weight
[2024-10-16 15:43:14.310234][INFO][hf2megads_weight_converter.py:113] - model.layers.19.post_attention_layernorm.weight
[2024-10-16 15:43:14.312927][INFO][hf2megads_weight_converter.py:113] - model.layers.20.self_attn.q_proj.weight
[2024-10-16 15:43:14.314505][INFO][hf2megads_weight_converter.py:113] - model.layers.20.self_attn.k_proj.weight
[2024-10-16 15:43:14.315992][INFO][hf2megads_weight_converter.py:113] - model.layers.20.self_attn.v_proj.weight
[2024-10-16 15:43:14.318788][INFO][hf2megads_weight_converter.py:113] - model.layers.20.self_attn.o_proj.weight
[2024-10-16 15:43:14.325390][INFO][hf2megads_weight_converter.py:113] - model.layers.20.mlp.gate_proj.weight
[2024-10-16 15:43:14.332020][INFO][hf2megads_weight_converter.py:113] - model.layers.20.mlp.up_proj.weight
[2024-10-16 15:43:14.338682][INFO][hf2megads_weight_converter.py:113] - model.layers.20.mlp.down_proj.weight
[2024-10-16 15:43:14.339334][INFO][hf2megads_weight_converter.py:113] - model.layers.20.input_layernorm.weight
[2024-10-16 15:43:14.339845][INFO][hf2megads_weight_converter.py:113] - model.layers.20.post_attention_layernorm.weight
[2024-10-16 15:43:14.342562][INFO][hf2megads_weight_converter.py:113] - model.layers.21.self_attn.q_proj.weight
[2024-10-16 15:43:14.344113][INFO][hf2megads_weight_converter.py:113] - model.layers.21.self_attn.k_proj.weight
[2024-10-16 15:43:14.345593][INFO][hf2megads_weight_converter.py:113] - model.layers.21.self_attn.v_proj.weight
[2024-10-16 15:43:14.348370][INFO][hf2megads_weight_converter.py:113] - model.layers.21.self_attn.o_proj.weight
[2024-10-16 15:43:14.355167][INFO][hf2megads_weight_converter.py:113] - model.layers.21.mlp.gate_proj.weight
[2024-10-16 15:43:14.361823][INFO][hf2megads_weight_converter.py:113] - model.layers.21.mlp.up_proj.weight
[2024-10-16 15:43:14.368428][INFO][hf2megads_weight_converter.py:113] - model.layers.21.mlp.down_proj.weight
[2024-10-16 15:43:14.369055][INFO][hf2megads_weight_converter.py:113] - model.layers.21.input_layernorm.weight
[2024-10-16 15:43:14.369558][INFO][hf2megads_weight_converter.py:113] - model.layers.21.post_attention_layernorm.weight
[2024-10-16 15:43:14.372269][INFO][hf2megads_weight_converter.py:113] - model.layers.22.self_attn.q_proj.weight
[2024-10-16 15:43:14.373830][INFO][hf2megads_weight_converter.py:113] - model.layers.22.self_attn.k_proj.weight
[2024-10-16 15:43:14.375316][INFO][hf2megads_weight_converter.py:113] - model.layers.22.self_attn.v_proj.weight
[2024-10-16 15:43:14.378084][INFO][hf2megads_weight_converter.py:113] - model.layers.22.self_attn.o_proj.weight
[2024-10-16 15:43:14.384700][INFO][hf2megads_weight_converter.py:113] - model.layers.22.mlp.gate_proj.weight
[2024-10-16 15:43:14.391366][INFO][hf2megads_weight_converter.py:113] - model.layers.22.mlp.up_proj.weight
[2024-10-16 15:43:14.398053][INFO][hf2megads_weight_converter.py:113] - model.layers.22.mlp.down_proj.weight
[2024-10-16 15:43:14.398695][INFO][hf2megads_weight_converter.py:113] - model.layers.22.input_layernorm.weight
[2024-10-16 15:43:14.399206][INFO][hf2megads_weight_converter.py:113] - model.layers.22.post_attention_layernorm.weight
[2024-10-16 15:43:14.401929][INFO][hf2megads_weight_converter.py:113] - model.layers.23.self_attn.q_proj.weight
[2024-10-16 15:43:14.403487][INFO][hf2megads_weight_converter.py:113] - model.layers.23.self_attn.k_proj.weight
[2024-10-16 15:43:14.404961][INFO][hf2megads_weight_converter.py:113] - model.layers.23.self_attn.v_proj.weight
[2024-10-16 15:43:14.407720][INFO][hf2megads_weight_converter.py:113] - model.layers.23.self_attn.o_proj.weight
[2024-10-16 15:43:14.414356][INFO][hf2megads_weight_converter.py:113] - model.layers.23.mlp.gate_proj.weight
[2024-10-16 15:43:14.421000][INFO][hf2megads_weight_converter.py:113] - model.layers.23.mlp.up_proj.weight
[2024-10-16 15:43:14.427610][INFO][hf2megads_weight_converter.py:113] - model.layers.23.mlp.down_proj.weight
[2024-10-16 15:43:14.428269][INFO][hf2megads_weight_converter.py:113] - model.layers.23.input_layernorm.weight
[2024-10-16 15:43:14.428783][INFO][hf2megads_weight_converter.py:113] - model.layers.23.post_attention_layernorm.weight
[2024-10-16 15:43:14.431461][INFO][hf2megads_weight_converter.py:113] - model.layers.24.self_attn.q_proj.weight
[2024-10-16 15:43:14.432995][INFO][hf2megads_weight_converter.py:113] - model.layers.24.self_attn.k_proj.weight
[2024-10-16 15:43:14.434507][INFO][hf2megads_weight_converter.py:113] - model.layers.24.self_attn.v_proj.weight
[2024-10-16 15:43:14.437268][INFO][hf2megads_weight_converter.py:113] - model.layers.24.self_attn.o_proj.weight
[2024-10-16 15:43:14.443850][INFO][hf2megads_weight_converter.py:113] - model.layers.24.mlp.gate_proj.weight
[2024-10-16 15:43:14.450632][INFO][hf2megads_weight_converter.py:113] - model.layers.24.mlp.up_proj.weight
[2024-10-16 15:43:14.457242][INFO][hf2megads_weight_converter.py:113] - model.layers.24.mlp.down_proj.weight
[2024-10-16 15:43:14.457890][INFO][hf2megads_weight_converter.py:113] - model.layers.24.input_layernorm.weight
[2024-10-16 15:43:14.458409][INFO][hf2megads_weight_converter.py:113] - model.layers.24.post_attention_layernorm.weight
[2024-10-16 15:43:14.461063][INFO][hf2megads_weight_converter.py:113] - model.layers.25.self_attn.q_proj.weight
[2024-10-16 15:43:14.462620][INFO][hf2megads_weight_converter.py:113] - model.layers.25.self_attn.k_proj.weight
[2024-10-16 15:43:14.464102][INFO][hf2megads_weight_converter.py:113] - model.layers.25.self_attn.v_proj.weight
[2024-10-16 15:43:14.466871][INFO][hf2megads_weight_converter.py:113] - model.layers.25.self_attn.o_proj.weight
[2024-10-16 15:43:14.473435][INFO][hf2megads_weight_converter.py:113] - model.layers.25.mlp.gate_proj.weight
[2024-10-16 15:43:14.480017][INFO][hf2megads_weight_converter.py:113] - model.layers.25.mlp.up_proj.weight
[2024-10-16 15:43:14.486605][INFO][hf2megads_weight_converter.py:113] - model.layers.25.mlp.down_proj.weight
[2024-10-16 15:43:14.487227][INFO][hf2megads_weight_converter.py:113] - model.layers.25.input_layernorm.weight
[2024-10-16 15:43:14.487743][INFO][hf2megads_weight_converter.py:113] - model.layers.25.post_attention_layernorm.weight
[2024-10-16 15:43:14.490427][INFO][hf2megads_weight_converter.py:113] - model.layers.26.self_attn.q_proj.weight
[2024-10-16 15:43:14.491946][INFO][hf2megads_weight_converter.py:113] - model.layers.26.self_attn.k_proj.weight
[2024-10-16 15:43:14.493433][INFO][hf2megads_weight_converter.py:113] - model.layers.26.self_attn.v_proj.weight
[2024-10-16 15:43:14.496192][INFO][hf2megads_weight_converter.py:113] - model.layers.26.self_attn.o_proj.weight
[2024-10-16 15:43:14.502792][INFO][hf2megads_weight_converter.py:113] - model.layers.26.mlp.gate_proj.weight
[2024-10-16 15:43:14.509329][INFO][hf2megads_weight_converter.py:113] - model.layers.26.mlp.up_proj.weight
[2024-10-16 15:43:14.515980][INFO][hf2megads_weight_converter.py:113] - model.layers.26.mlp.down_proj.weight
[2024-10-16 15:43:14.516659][INFO][hf2megads_weight_converter.py:113] - model.layers.26.input_layernorm.weight
[2024-10-16 15:43:14.517200][INFO][hf2megads_weight_converter.py:113] - model.layers.26.post_attention_layernorm.weight
[2024-10-16 15:43:14.519874][INFO][hf2megads_weight_converter.py:113] - model.layers.27.self_attn.q_proj.weight
[2024-10-16 15:43:14.521415][INFO][hf2megads_weight_converter.py:113] - model.layers.27.self_attn.k_proj.weight
[2024-10-16 15:43:14.522879][INFO][hf2megads_weight_converter.py:113] - model.layers.27.self_attn.v_proj.weight
[2024-10-16 15:43:14.525620][INFO][hf2megads_weight_converter.py:113] - model.layers.27.self_attn.o_proj.weight
[2024-10-16 15:43:14.532202][INFO][hf2megads_weight_converter.py:113] - model.layers.27.mlp.gate_proj.weight
[2024-10-16 15:43:14.538768][INFO][hf2megads_weight_converter.py:113] - model.layers.27.mlp.up_proj.weight
[2024-10-16 15:43:14.545303][INFO][hf2megads_weight_converter.py:113] - model.layers.27.mlp.down_proj.weight
[2024-10-16 15:43:14.545921][INFO][hf2megads_weight_converter.py:113] - model.layers.27.input_layernorm.weight
[2024-10-16 15:43:14.546440][INFO][hf2megads_weight_converter.py:113] - model.layers.27.post_attention_layernorm.weight
[2024-10-16 15:43:14.549101][INFO][hf2megads_weight_converter.py:113] - model.layers.28.self_attn.q_proj.weight
[2024-10-16 15:43:14.550596][INFO][hf2megads_weight_converter.py:113] - model.layers.28.self_attn.k_proj.weight
[2024-10-16 15:43:14.552114][INFO][hf2megads_weight_converter.py:113] - model.layers.28.self_attn.v_proj.weight
[2024-10-16 15:43:14.554821][INFO][hf2megads_weight_converter.py:113] - model.layers.28.self_attn.o_proj.weight
[2024-10-16 15:43:14.561373][INFO][hf2megads_weight_converter.py:113] - model.layers.28.mlp.gate_proj.weight
[2024-10-16 15:43:14.567945][INFO][hf2megads_weight_converter.py:113] - model.layers.28.mlp.up_proj.weight
[2024-10-16 15:43:14.574713][INFO][hf2megads_weight_converter.py:113] - model.layers.28.mlp.down_proj.weight
[2024-10-16 15:43:14.575333][INFO][hf2megads_weight_converter.py:113] - model.layers.28.input_layernorm.weight
[2024-10-16 15:43:14.575833][INFO][hf2megads_weight_converter.py:113] - model.layers.28.post_attention_layernorm.weight
[2024-10-16 15:43:14.578469][INFO][hf2megads_weight_converter.py:113] - model.layers.29.self_attn.q_proj.weight
[2024-10-16 15:43:14.580020][INFO][hf2megads_weight_converter.py:113] - model.layers.29.self_attn.k_proj.weight
[2024-10-16 15:43:14.581498][INFO][hf2megads_weight_converter.py:113] - model.layers.29.self_attn.v_proj.weight
[2024-10-16 15:43:14.584217][INFO][hf2megads_weight_converter.py:113] - model.layers.29.self_attn.o_proj.weight
[2024-10-16 15:43:14.590850][INFO][hf2megads_weight_converter.py:113] - model.layers.29.mlp.gate_proj.weight
[2024-10-16 15:43:14.597375][INFO][hf2megads_weight_converter.py:113] - model.layers.29.mlp.up_proj.weight
[2024-10-16 15:43:14.603899][INFO][hf2megads_weight_converter.py:113] - model.layers.29.mlp.down_proj.weight
[2024-10-16 15:43:14.604548][INFO][hf2megads_weight_converter.py:113] - model.layers.29.input_layernorm.weight
[2024-10-16 15:43:14.605067][INFO][hf2megads_weight_converter.py:113] - model.layers.29.post_attention_layernorm.weight
[2024-10-16 15:43:14.607694][INFO][hf2megads_weight_converter.py:113] - model.layers.30.self_attn.q_proj.weight
[2024-10-16 15:43:14.609232][INFO][hf2megads_weight_converter.py:113] - model.layers.30.self_attn.k_proj.weight
[2024-10-16 15:43:14.610733][INFO][hf2megads_weight_converter.py:113] - model.layers.30.self_attn.v_proj.weight
[2024-10-16 15:43:14.613448][INFO][hf2megads_weight_converter.py:113] - model.layers.30.self_attn.o_proj.weight
[2024-10-16 15:43:14.619991][INFO][hf2megads_weight_converter.py:113] - model.layers.30.mlp.gate_proj.weight
[2024-10-16 15:43:14.626559][INFO][hf2megads_weight_converter.py:113] - model.layers.30.mlp.up_proj.weight
[2024-10-16 15:43:14.633070][INFO][hf2megads_weight_converter.py:113] - model.layers.30.mlp.down_proj.weight
[2024-10-16 15:43:14.633733][INFO][hf2megads_weight_converter.py:113] - model.layers.30.input_layernorm.weight
[2024-10-16 15:43:14.634259][INFO][hf2megads_weight_converter.py:113] - model.layers.30.post_attention_layernorm.weight
[2024-10-16 15:43:14.636877][INFO][hf2megads_weight_converter.py:113] - model.layers.31.self_attn.q_proj.weight
[2024-10-16 15:43:14.638487][INFO][hf2megads_weight_converter.py:113] - model.layers.31.self_attn.k_proj.weight
[2024-10-16 15:43:14.639954][INFO][hf2megads_weight_converter.py:113] - model.layers.31.self_attn.v_proj.weight
[2024-10-16 15:43:14.642672][INFO][hf2megads_weight_converter.py:113] - model.layers.31.self_attn.o_proj.weight
[2024-10-16 15:43:14.649190][INFO][hf2megads_weight_converter.py:113] - model.layers.31.mlp.gate_proj.weight
[2024-10-16 15:43:14.655715][INFO][hf2megads_weight_converter.py:113] - model.layers.31.mlp.up_proj.weight
[2024-10-16 15:43:14.662234][INFO][hf2megads_weight_converter.py:113] - model.layers.31.mlp.down_proj.weight
[2024-10-16 15:43:14.662858][INFO][hf2megads_weight_converter.py:113] - model.layers.31.input_layernorm.weight
[2024-10-16 15:43:14.663389][INFO][hf2megads_weight_converter.py:113] - model.layers.31.post_attention_layernorm.weight
[2024-10-16 15:43:14.663915][INFO][hf2megads_weight_converter.py:113] - model.norm.weight
[2024-10-16 15:43:14,693] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
[2024-10-16 15:43:14.713565][INFO][hf2megads_weight_converter.py:113] - lm_head.weight
[2024-10-16 15:43:14.714574][INFO][hf2megads_weight_converter.py:504] - before deepspeed init
[2024-10-16 15:43:14,715] [INFO] [logging.py:129:log_dist] [Rank 0] DeepSpeed info: version=0.15.3+unknown, git-hash=unknown, git-branch=unknown
[2024-10-16 15:43:14,715] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
Loading checkpoint shards: 100%|██████████| 7/7 [03:57<00:00, 33.97s/it]
[2024-10-16 15:43:36,676] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
Loading checkpoint shards: 100%|██████████| 7/7 [04:00<00:00, 34.36s/it]
Loading checkpoint shards: 100%|██████████| 7/7 [04:00<00:00, 34.38s/it]
Loading checkpoint shards: 100%|██████████| 7/7 [04:00<00:00, 34.40s/it]
Loading checkpoint shards: 100%|██████████| 7/7 [04:00<00:00, 34.39s/it]
Loading checkpoint shards: 100%|██████████| 7/7 [04:00<00:00, 34.41s/it]
Loading checkpoint shards: 100%|██████████| 7/7 [04:01<00:00, 34.43s/it]
Loading checkpoint shards: 100%|██████████| 7/7 [04:00<00:00, 34.40s/it]
Loading checkpoint shards: 100%|██████████| 7/7 [04:00<00:00, 34.41s/it]
Loading checkpoint shards: 100%|██████████| 7/7 [04:00<00:00, 34.41s/it]
Loading checkpoint shards: 100%|██████████| 7/7 [04:01<00:00, 34.44s/it]
Loading checkpoint shards: 100%|██████████| 7/7 [04:01<00:00, 34.44s/it]
Loading checkpoint shards: 100%|██████████| 7/7 [04:01<00:00, 34.46s/it]
Loading checkpoint shards: 100%|██████████| 7/7 [04:01<00:00, 34.44s/it]
Loading checkpoint shards: 100%|██████████| 7/7 [04:01<00:00, 34.43s/it]
Loading checkpoint shards: 100%|██████████| 7/7 [04:01<00:00, 34.47s/it]
Loading checkpoint shards: 100%|██████████| 7/7 [04:01<00:00, 34.44s/it]
Loading checkpoint shards: 100%|██████████| 7/7 [04:01<00:00, 34.47s/it]
Loading checkpoint shards: 100%|██████████| 7/7 [04:01<00:00, 34.44s/it]
Loading checkpoint shards: 100%|██████████| 7/7 [03:58<00:00, 34.12s/it]
Loading checkpoint shards: 100%|██████████| 7/7 [04:01<00:00, 34.47s/it]
Loading checkpoint shards: 100%|██████████| 7/7 [04:01<00:00, 34.51s/it]
[2024-10-16 15:43:39,455] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
[2024-10-16 15:43:39,666] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
[2024-10-16 15:43:39,739] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
[2024-10-16 15:43:39,801] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
[2024-10-16 15:43:39,842] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
[2024-10-16 15:43:39,860] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
[2024-10-16 15:43:39,862] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
[2024-10-16 15:43:39,881] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
[2024-10-16 15:43:39,920] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
[2024-10-16 15:43:40,138] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
[2024-10-16 15:43:40,153] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
[2024-10-16 15:43:40,174] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
[2024-10-16 15:43:40,175] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
[2024-10-16 15:43:40,205] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
[2024-10-16 15:43:40,212] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
[2024-10-16 15:43:40,224] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
[2024-10-16 15:43:40,251] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
[2024-10-16 15:43:40,255] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
[2024-10-16 15:43:40,256] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
[2024-10-16 15:43:40,417] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
[2024-10-16 15:43:40,538] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 24
[2024-10-16 15:43:56,920] [INFO] [logging.py:129:log_dist] [Rank 0] DeepSpeed Flops Profiler Enabled: False
[2024-10-16 15:43:56,921] [INFO] [logging.py:129:log_dist] [Rank 0] Creating BF16 optimizer
[2024-10-16 15:43:57,118] [INFO] [utils.py:781:see_memory_usage] begin bf16_optimizer
[2024-10-16 15:43:57,118] [INFO] [utils.py:782:see_memory_usage] MA 14.96 GB         Max_MA 14.96 GB         CA 14.96 GB         Max_CA 15 GB
[2024-10-16 15:43:57,118] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 490.74 GB, percent = 43.3%
[2024-10-16 15:43:57,290] [INFO] [utils.py:781:see_memory_usage] end bf16_ optimizer
[2024-10-16 15:43:57,291] [INFO] [utils.py:782:see_memory_usage] MA 14.96 GB         Max_MA 14.96 GB         CA 14.96 GB         Max_CA 15 GB
[2024-10-16 15:43:57,291] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 490.74 GB, percent = 43.3%
[2024-10-16 15:43:57,291] [INFO] [config.py:999:print] DeepSpeedEngine configuration:
[2024-10-16 15:43:57,292] [INFO] [config.py:1003:print]   activation_checkpointing_config  {
    "partition_activations": false,
    "contiguous_memory_optimization": false,
    "cpu_checkpointing": false,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false,
    "profile": false
}
[2024-10-16 15:43:57,292] [INFO] [config.py:1003:print]   aio_config ................... {'block_size': 1048576, 'queue_depth': 8, 'thread_count': 1, 'single_submit': False, 'overlap_events': True, 'use_gds': False}
[2024-10-16 15:43:57,292] [INFO] [config.py:1003:print]   amp_enabled .................. False
[2024-10-16 15:43:57,292] [INFO] [config.py:1003:print]   amp_params ................... False
[2024-10-16 15:43:57,292] [INFO] [config.py:1003:print]   autotuning_config ............ {
    "enabled": false,
    "start_step": null,
    "end_step": null,
    "metric_path": null,
    "arg_mappings": null,
    "metric": "throughput",
    "model_info": null,
    "results_dir": "autotuning_results",
    "exps_dir": "autotuning_exps",
    "overwrite": true,
    "fast": true,
    "start_profile_step": 3,
    "end_profile_step": 5,
    "tuner_type": "gridsearch",
    "tuner_early_stopping": 5,
    "tuner_num_trials": 50,
    "model_info_path": null,
    "mp_size": 1,
    "max_train_batch_size": null,
    "min_train_batch_size": 1,
    "max_train_micro_batch_size_per_gpu": 1.024000e+03,
    "min_train_micro_batch_size_per_gpu": 1,
    "num_tuning_micro_batch_sizes": 3
}
[2024-10-16 15:43:57,292] [INFO] [config.py:1003:print]   bfloat16_enabled ............. True
[2024-10-16 15:43:57,292] [INFO] [config.py:1003:print]   bfloat16_immediate_grad_update  False
[2024-10-16 15:43:57,292] [INFO] [config.py:1003:print]   checkpoint_parallel_write_pipeline  False
[2024-10-16 15:43:57,292] [INFO] [config.py:1003:print]   checkpoint_tag_validation_enabled  True
[2024-10-16 15:43:57,292] [INFO] [config.py:1003:print]   checkpoint_tag_validation_fail  False
[2024-10-16 15:43:57,292] [INFO] [config.py:1003:print]   comms_config ................. <deepspeed.comm.config.DeepSpeedCommsConfig object at 0x145ea0815900>
[2024-10-16 15:43:57,292] [INFO] [config.py:1003:print]   communication_data_type ...... None
[2024-10-16 15:43:57,292] [INFO] [config.py:1003:print]   compression_config ........... {'weight_quantization': {'shared_parameters': {'enabled': False, 'quantizer_kernel': False, 'schedule_offset': 0, 'quantize_groups': 1, 'quantize_verbose': False, 'quantization_type': 'symmetric', 'quantize_weight_in_forward': False, 'rounding': 'nearest', 'fp16_mixed_quantize': False, 'quantize_change_ratio': 0.001}, 'different_groups': {}}, 'activation_quantization': {'shared_parameters': {'enabled': False, 'quantization_type': 'symmetric', 'range_calibration': 'dynamic', 'schedule_offset': 1000}, 'different_groups': {}}, 'sparse_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'row_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'head_pruning': {'shared_parameters': {'enabled': False, 'method': 'topk', 'schedule_offset': 1000}, 'different_groups': {}}, 'channel_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'layer_reduction': {'enabled': False}}
[2024-10-16 15:43:57,292] [INFO] [config.py:1003:print]   curriculum_enabled_legacy .... False
[2024-10-16 15:43:57,292] [INFO] [config.py:1003:print]   curriculum_params_legacy ..... False
[2024-10-16 15:43:57,292] [INFO] [config.py:1003:print]   data_efficiency_config ....... {'enabled': False, 'seed': 1234, 'data_sampling': {'enabled': False, 'num_epochs': 1000, 'num_workers': 0, 'curriculum_learning': {'enabled': False}}, 'data_routing': {'enabled': False, 'random_ltd': {'enabled': False, 'layer_token_lr_schedule': {'enabled': False}}}}
[2024-10-16 15:43:57,292] [INFO] [config.py:1003:print]   data_efficiency_enabled ...... False
[2024-10-16 15:43:57,292] [INFO] [config.py:1003:print]   dataloader_drop_last ......... False
[2024-10-16 15:43:57,292] [INFO] [config.py:1003:print]   disable_allgather ............ False
[2024-10-16 15:43:57,292] [INFO] [config.py:1003:print]   dump_state ................... False
[2024-10-16 15:43:57,292] [INFO] [config.py:1003:print]   dynamic_loss_scale_args ...... None
[2024-10-16 15:43:57,292] [INFO] [config.py:1003:print]   eigenvalue_enabled ........... False
[2024-10-16 15:43:57,292] [INFO] [config.py:1003:print]   eigenvalue_gas_boundary_resolution  1
[2024-10-16 15:43:57,292] [INFO] [config.py:1003:print]   eigenvalue_layer_name ........ bert.encoder.layer
[2024-10-16 15:43:57,292] [INFO] [config.py:1003:print]   eigenvalue_layer_num ......... 0
[2024-10-16 15:43:57,292] [INFO] [config.py:1003:print]   eigenvalue_max_iter .......... 100
[2024-10-16 15:43:57,292] [INFO] [config.py:1003:print]   eigenvalue_stability ......... 1e-06
[2024-10-16 15:43:57,292] [INFO] [config.py:1003:print]   eigenvalue_tol ............... 0.01
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   eigenvalue_verbose ........... False
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   elasticity_enabled ........... False
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   flops_profiler_config ........ {
    "enabled": false,
    "recompute_fwd_factor": 0.0,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 1,
    "detailed": true,
    "output_file": null
}
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   fp16_auto_cast ............... None
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   fp16_enabled ................. False
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   fp16_master_weights_and_gradients  False
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   global_rank .................. 0
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   grad_accum_dtype ............. None
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   gradient_accumulation_steps .. 1
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   gradient_clipping ............ 0.0
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   gradient_predivide_factor .... 1.0
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   graph_harvesting ............. False
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   hybrid_engine ................ enabled=False max_out_tokens=512 inference_tp_size=1 release_inference_cache=False pin_parameters=True tp_gather_partition_size=8
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   initial_dynamic_scale ........ 1
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   load_universal_checkpoint .... False
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   loss_scale ................... 1.0
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   memory_breakdown ............. False
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   mics_hierarchial_params_gather  False
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   mics_shard_size .............. -1
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   monitor_config ............... tensorboard=TensorBoardConfig(enabled=False, output_path='', job_name='DeepSpeedJobName') comet=CometConfig(enabled=False, samples_log_interval=100, project=None, workspace=None, api_key=None, experiment_name=None, experiment_key=None, online=None, mode=None) wandb=WandbConfig(enabled=False, group=None, team=None, project='deepspeed') csv_monitor=CSVConfig(enabled=False, output_path='', job_name='DeepSpeedJobName')
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   nebula_config ................ {
    "enabled": false,
    "persistent_storage_path": null,
    "persistent_time_interval": 100,
    "num_of_version_in_retention": 2,
    "enable_nebula_load": true,
    "load_path": null
}
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   optimizer_legacy_fusion ...... False
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   optimizer_name ............... None
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   optimizer_params ............. None
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   pipeline ..................... {'stages': 'auto', 'partition': 'best', 'seed_layers': False, 'activation_checkpoint_interval': 0, 'pipe_partitioned': True, 'grad_partitioned': True}
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   pld_enabled .................. False
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   pld_params ................... False
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   prescale_gradients ........... False
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   scheduler_name ............... None
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   scheduler_params ............. None
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   seq_parallel_communication_data_type  torch.float32
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   sparse_attention ............. None
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   sparse_gradients_enabled ..... False
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   steps_per_print .............. 100
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   timers_config ................ enabled=True synchronized=True
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   train_batch_size ............. 24
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   train_micro_batch_size_per_gpu  1
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   use_data_before_expert_parallel_  False
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   use_node_local_storage ....... False
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   wall_clock_breakdown ......... False
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   weight_quantization_config ... None
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   world_size ................... 24
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   zero_allow_untested_optimizer  False
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   zero_config .................. stage=0 contiguous_gradients=True reduce_scatter=True reduce_bucket_size=500000000 use_multi_rank_bucket_allreduce=True allgather_partitions=True allgather_bucket_size=500000000 overlap_comm=False load_from_fp32_weights=True elastic_checkpoint=False offload_param=None offload_optimizer=None sub_group_size=1000000000 cpu_offload_param=None cpu_offload_use_pin_memory=None cpu_offload=None prefetch_bucket_size=50000000 param_persistence_threshold=100000 model_persistence_threshold=9223372036854775807 max_live_parameters=1000000000 max_reuse_distance=1000000000 gather_16bit_weights_on_model_save=False use_all_reduce_for_fetch_params=False stage3_gather_fp16_weights_on_model_save=False ignore_unused_parameters=True legacy_stage1=False round_robin_gradients=False zero_hpz_partition_size=1 zero_quantized_weights=False zero_quantized_nontrainable_weights=False zero_quantized_gradients=False mics_shard_size=-1 mics_hierarchical_params_gather=False memory_efficient_linear=True pipeline_loading_checkpoint=False override_module_apply=True
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   zero_enabled ................. False
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   zero_force_ds_cpu_optimizer .. True
[2024-10-16 15:43:57,293] [INFO] [config.py:1003:print]   zero_optimization_stage ...... 0
[2024-10-16 15:43:57,293] [INFO] [config.py:989:print_user_config]   json = {
    "train_batch_size": 24,
    "train_micro_batch_size_per_gpu": 1,
    "steps_per_print": 100,
    "zero_optimization": {
        "stage": 0
    },
    "bf16": {
        "enabled": true
    }
}
[2024-10-16 15:43:57,293] [INFO] [engine.py:105:__init__] CONFIG: micro_batches=1 micro_batch_size=1
[2024-10-16 15:43:57,294] [INFO] [engine.py:146:__init__] is_pipe_partitioned= False is_grad_partitioned= False
[2024-10-16 15:43:57,298] [INFO] [engine.py:165:__init__] RANK=0 STAGE=0 LAYERS=37 [0, 37) STAGE_PARAMS=8032358400 (8032.358M) TOTAL_PARAMS=8032358400 (8032.358M) UNIQUE_PARAMS=8032358400 (8032.358M)
[2024-10-16 15:43:57.298751][INFO][hf2megads_weight_converter.py:511] - after deepspeed init
[2024-10-16 15:43:57.299527][INFO][hf2megads_weight_converter.py:162] - hf_w.shape[0]=128512
[2024-10-16 15:43:57.299951][INFO][hf2megads_weight_converter.py:163] - self.token_vocab=128000
[2024-10-16 15:43:57,373] [INFO] [engine.py:146:__init__] is_pipe_partitioned= False is_grad_partitioned= False
[2024-10-16 15:43:57,410] [INFO] [engine.py:146:__init__] is_pipe_partitioned= False is_grad_partitioned= False
[2024-10-16 15:43:57,525] [INFO] [engine.py:146:__init__] is_pipe_partitioned= False is_grad_partitioned= False
                                     [--hidden-size HIDDEN_SIZE]
!!! ATTENTION !!!
                                     [--num-attention-heads NUM_ATTENTION_HEADS]
Type 'up' to get to the frame that called dist.breakpoint(rank=0)
                                     [--kv-channels KV_CHANNELS]
> /opt/aurora/24.180.0/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages/torch/distributed/__init__.py(89)breakpoint()
-> barrier()
(Pdb) [2024-10-16 15:43:57,531] [INFO] [engine.py:146:__init__] is_pipe_partitioned= False is_grad_partitioned= False
[2024-10-16 15:43:57,539] [INFO] [engine.py:146:__init__] is_pipe_partitioned= False is_grad_partitioned= False
[2024-10-16 15:43:57,613] [INFO] [engine.py:146:__init__] is_pipe_partitioned= False is_grad_partitioned= False
[2024-10-16 15:43:57,677] [INFO] [engine.py:146:__init__] is_pipe_partitioned= False is_grad_partitioned= False
[2024-10-16 15:43:57,689] [INFO] [engine.py:146:__init__] is_pipe_partitioned= False is_grad_partitioned= False
[2024-10-16 15:43:57,689] [INFO] [engine.py:146:__init__] is_pipe_partitioned= False is_grad_partitioned= False
[2024-10-16 15:43:57,694] [INFO] [engine.py:146:__init__] is_pipe_partitioned= False is_grad_partitioned= False
[2024-10-16 15:43:57,694] [INFO] [engine.py:146:__init__] is_pipe_partitioned= False is_grad_partitioned= False
[2024-10-16 15:43:57,733] [INFO] [engine.py:146:__init__] is_pipe_partitioned= False is_grad_partitioned= False
[2024-10-16 15:43:57,768] [INFO] [engine.py:146:__init__] is_pipe_partitioned= False is_grad_partitioned= False
[2024-10-16 15:43:57,769] [INFO] [engine.py:146:__init__] is_pipe_partitioned= False is_grad_partitioned= False
[2024-10-16 15:43:57,823] [INFO] [engine.py:146:__init__] is_pipe_partitioned= False is_grad_partitioned= False
[2024-10-16 15:43:57,857] [INFO] [engine.py:146:__init__] is_pipe_partitioned= False is_grad_partitioned= False
[2024-10-16 15:43:57,869] [INFO] [engine.py:146:__init__] is_pipe_partitioned= False is_grad_partitioned= False
[2024-10-16 15:43:57,876] [INFO] [engine.py:146:__init__] is_pipe_partitioned= False is_grad_partitioned= False
[2024-10-16 15:43:57,876] [INFO] [engine.py:146:__init__] is_pipe_partitioned= False is_grad_partitioned= False
[2024-10-16 15:43:58,057] [INFO] [engine.py:146:__init__] is_pipe_partitioned= False is_grad_partitioned= False
[2024-10-16 15:43:58,058] [INFO] [engine.py:146:__init__] is_pipe_partitioned= False is_grad_partitioned= False
[2024-10-16 15:43:58,102] [INFO] [engine.py:146:__init__] is_pipe_partitioned= False is_grad_partitioned= False
[2024-10-16 15:43:58,102] [INFO] [engine.py:146:__init__] is_pipe_partitioned= False is_grad_partitioned= False
                                     [--ffn-hidden-size FFN_HIDDEN_SIZE]
(Pdb) l
 84                pdb.message(
 85                    "\n!!! ATTENTION !!!\n\n"
 86                    f"Type 'up' to get to the frame that called dist.breakpoint(rank={rank})\n"
 87                )
 88                pdb.set_trace()
 89  ->        barrier()
 90     
 91        if sys.platform != "win32":
 92            from torch._C._distributed_c10d import (
 93                HashStore,
 94                _round_robin_process_groups,
(Pdb) ll
 74        def breakpoint(rank: int = 0):
 75            """
 76            Set a breakpoint, but only on a single rank.  All other ranks will wait for you to be
 77            done with the breakpoint before continuing.
 78     
 79            Args:
 80                rank (int): Which rank to break on.  Default: ``0``
 81            """
 82            if get_rank() == rank:
 83                pdb = _DistributedPdb()
 84                pdb.message(
 85                    "\n!!! ATTENTION !!!\n\n"
 86                    f"Type 'up' to get to the frame that called dist.breakpoint(rank={rank})\n"
 87                )
 88                pdb.set_trace()
 89  ->        barrier()
(Pdb) up
> /lus/flare/projects/Aurora_deployment/foremans/projects/argonne-lcf/Megatron-DeepSpeed/tools/hf2megads_weight_converter.py(224)_qkv_refactor()
-> torch.distributed.breakpoint(0)
(Pdb) ll
194        def _qkv_refactor(self, pname, p, hf_layer):
195            hf_wq_name = f"model.layers.{hf_layer}.self_attn.q_proj.weight"
196            hf_wk_name = f"model.layers.{hf_layer}.self_attn.k_proj.weight"
197            hf_wv_name = f"model.layers.{hf_layer}.self_attn.v_proj.weight"
198            wq = self.hf_model[hf_wq_name]
199            wk = self.hf_model[hf_wk_name]
200            wv = self.hf_model[hf_wv_name]
201     
202            hidden_size = wq.shape[0]
203            per_partition_size, start_index, end_index = compute_partition_range(
204                hidden_size, self.tp_rank, self.tp_size)
205            hidden_size_per_attention_head = divide(hidden_size,
206                                                    self.config.num_attention_heads)
207            num_attention_heads_per_partition = divide(self.config.num_attention_heads,
208                                                       self.tp_size)
209     
210            new_w = torch.zeros((per_partition_size * 3, wq.shape[1]), dtype=wq.dtype)
211     
212            for i in range(num_attention_heads_per_partition):
213                try:
214                    current_index = start_index + i * hidden_size_per_attention_head
215                    next_index = current_index + hidden_size_per_attention_head
216                    new_w_index = i * (3 * hidden_size_per_attention_head)
217                    new_w[new_w_index: new_w_index + (3 * hidden_size_per_attention_head), :] = \
218                        torch.cat([
219                            wq[current_index: next_index, :],
220                            wk[current_index: next_index, :],
221                            wv[current_index: next_index, :]
222                        ], dim=0)
223                except Exception:
224  ->                torch.distributed.breakpoint(0)
225            self.record_mapping_info(
226                f"mega-ds:{pname,p.data.shape}<--hf{hf_wq_name,hf_wk_name,hf_wv_name,}  cat q,k,v [{current_index}:{next_index},:]  of q,k,v{wq.shape}"
227            )
228            return new_w
(Pdb) current_index
1024
(Pdb) next_index
1152
(Pdb) new_w_index
3072
(Pdb) new_w.shape
torch.Size([12288, 4096])
(Pdb) wq
tensor([[ 0.0053, -0.0291, -0.0058,  ...,  0.0095, -0.0420, -0.0272],
        [-0.0142, -0.0679, -0.0049,  ..., -0.0142, -0.0498,  0.0192],
        [-0.0162, -0.0393, -0.0026,  ...,  0.0115, -0.0126,  0.0071],
        ...,
        [-0.0039, -0.0393,  0.0806,  ...,  0.0061, -0.0013,  0.0023],
        [-0.0035, -0.0101,  0.0459,  ...,  0.0049, -0.0011,  0.0011],
        [-0.0018, -0.0153,  0.0347,  ...,  0.0110,  0.0004,  0.0044]],
       dtype=torch.bfloat16, grad_fn=<CloneBackward0>)
(Pdb) wq.shape
torch.Size([4096, 4096])
(Pdb) wk.shape
torch.Size([1024, 4096])
(Pdb) wv.shape
torch.Size([1024, 4096])
(Pdb) hidden_size
4096
(Pdb) per_partition_size
4096
(Pdb) num_attention_heads_per_partition
32
(Pdb) new_w.shape
torch.Size([12288, 4096])
(Pdb) new_w.shape
torch.Size([12288, 4096])
(Pdb)
```

</details>
