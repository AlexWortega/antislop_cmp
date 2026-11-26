#!/usr/bin/env bash
set -euo pipefail
cd /workspace/pilot/vendor/verl
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/uvcache
P=/workspace/pilot/vendor/verl/.venv/bin/python
"$P" /workspace/pilot/build_data.py
"$P" -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
 data.train_files=/workspace/pilot/data/train.parquet data.val_files=/workspace/pilot/data/val.parquet \
 data.train_batch_size=4 data.val_batch_size=4 data.max_prompt_length=512 data.max_response_length=256 \
 data.filter_overlong_prompts=True data.truncation=error data.shuffle=True \
 '+data.apply_chat_template_kwargs.enable_thinking=False' \
 actor_rollout_ref.model.path=/workspace/model actor_rollout_ref.model.lora_rank=32 actor_rollout_ref.model.lora_alpha=64 \
 'actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,in_proj_z,out_proj]' \
 'actor_rollout_ref.model.exclude_modules=.*visual.*' \
 actor_rollout_ref.model.use_remove_padding=False actor_rollout_ref.model.enable_gradient_checkpointing=True \
 +actor_rollout_ref.actor.checkpoint.save_lora_only=True actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.actor.optim.lr=1e-5 \
 actor_rollout_ref.actor.ppo_mini_batch_size=4 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.actor.use_dynamic_bsz=False actor_rollout_ref.actor.use_torch_compile=False \
 actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.01 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 actor_rollout_ref.actor.fsdp_config.fsdp_size=1 actor_rollout_ref.actor.fsdp_config.param_offload=True \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=True actor_rollout_ref.actor.fsdp_config.offload_policy=False \
 actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
 actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 actor_rollout_ref.ref.strategy=fsdp2 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.ref.fsdp_config.param_offload=True actor_rollout_ref.ref.use_torch_compile=False \
 actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.5 actor_rollout_ref.rollout.n=4 \
 actor_rollout_ref.rollout.max_model_len=1024 actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
 actor_rollout_ref.rollout.max_num_seqs=16 actor_rollout_ref.rollout.enforce_eager=True \
 actor_rollout_ref.rollout.free_cache_engine=True actor_rollout_ref.rollout.enable_prefix_caching=False \
 actor_rollout_ref.rollout.layered_summon=True actor_rollout_ref.rollout.load_format=safetensors \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.rollout.val_kwargs.do_sample=False actor_rollout_ref.rollout.val_kwargs.n=1 \
 reward.num_workers=4 reward.custom_reward_function.path=/workspace/pilot/reward_fn.py \
 reward.custom_reward_function.name=compute_score \
 trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.total_training_steps=4 trainer.total_epochs=1 \
 trainer.val_before_train=False trainer.test_freq=4 trainer.save_freq=4 \
 'trainer.logger=[console]' trainer.project_name=qwen_antislop_pilot trainer.experiment_name=budget20 \
 trainer.default_local_dir=/workspace/pilot/output/checkpoints \
 trainer.rollout_data_dir=/workspace/pilot/output/rollouts trainer.validation_data_dir=/workspace/pilot/output/validation \
 ray_kwargs.ray_init.runtime_env.py_executable=null \
 "$@"
