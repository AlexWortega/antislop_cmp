#!/usr/bin/env bash
set -euo pipefail
cd /workspace/longrun/vendor/verl
export PYTHONPATH=/workspace/native:$PWD
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/uvcache
export VERL_FILE_LOGGER_PATH=/workspace/native/output/metrics.jsonl
export TENSORBOARD_DIR=/workspace/native/output/tensorboard
P=/workspace/longrun/vendor/verl/.venv/bin/python
"$P" -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
 data.train_files=/workspace/native/train.parquet data.val_files=/workspace/native/val.parquet \
 data.train_batch_size=8 data.val_batch_size=4 data.max_prompt_length=8192 data.max_response_length=8192 \
 data.filter_overlong_prompts=True data.truncation=error data.shuffle=False \
 '+data.apply_chat_template_kwargs.enable_thinking=False' \
 actor_rollout_ref.model.path=/workspace/antislop_model actor_rollout_ref.model.lora_rank=256 actor_rollout_ref.model.lora_alpha=512 \
 'actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,in_proj_z,out_proj]' \
 'actor_rollout_ref.model.exclude_modules=.*visual.*' \
 actor_rollout_ref.model.use_remove_padding=False actor_rollout_ref.model.enable_gradient_checkpointing=True \
 +actor_rollout_ref.actor.checkpoint.save_lora_only=True actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.actor.optim.lr=4e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=8 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.actor.use_dynamic_bsz=False actor_rollout_ref.actor.use_torch_compile=False \
 actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.01 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 actor_rollout_ref.actor.fsdp_config.fsdp_size=1 actor_rollout_ref.actor.fsdp_config.param_offload=True \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=True actor_rollout_ref.actor.fsdp_config.offload_policy=False \
 actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
 actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 actor_rollout_ref.ref.strategy=fsdp2 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.ref.fsdp_config.param_offload=True actor_rollout_ref.ref.use_torch_compile=False \
 actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.5 actor_rollout_ref.rollout.n=8 \
 actor_rollout_ref.rollout.max_model_len=65536 actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
 actor_rollout_ref.rollout.max_num_seqs=64 actor_rollout_ref.rollout.enforce_eager=True \
 actor_rollout_ref.rollout.free_cache_engine=True actor_rollout_ref.rollout.enable_prefix_caching=False \
 actor_rollout_ref.rollout.layered_summon=True actor_rollout_ref.rollout.load_format=safetensors \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.rollout.val_kwargs.do_sample=True actor_rollout_ref.rollout.val_kwargs.temperature=1.0 actor_rollout_ref.rollout.val_kwargs.n=4 \
 reward.num_workers=1 reward.custom_reward_function.path=/workspace/native/reward_fn.py \
 reward.custom_reward_function.name=compute_score \
 trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.total_training_steps=1 trainer.total_epochs=2 \
 trainer.val_before_train=False trainer.test_freq=-1 trainer.save_freq=1 \
 'trainer.logger=[console,file,tensorboard]' trainer.project_name=antislop_verl_native trainer.experiment_name=rank256-alpha512-8x8-fresh \
 trainer.default_local_dir=/workspace/native/output/checkpoints \
 trainer.rollout_data_dir=/workspace/native/output/rollouts trainer.validation_data_dir=/workspace/native/output/validation \
 ray_kwargs.ray_init.runtime_env.py_executable=null \
 trainer.use_v1=False \
 actor_rollout_ref.model.external_lib=native_compat \
 actor_rollout_ref.rollout.calculate_log_probs=True \
 actor_rollout_ref.rollout.agent.num_workers=1 \
 actor_rollout_ref.rollout.agent.default_agent_loop=harbor_native \
 actor_rollout_ref.rollout.agent.agent_loop_config_path=/workspace/native/agent.yaml \
 actor_rollout_ref.actor.ppo_epochs=1 \
 "$@"
