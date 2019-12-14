mkdir outputs
mkdir records

method=simple
gpu=5

mode=save_feat
model=ResNet18
optimizer=SGD

# stage-1
model_id=1
train_lr=0.01
resume=1
aug=0

save_freq=50
start_epoch=0
stop_epoch=200

vis_log=/home/gaojinghan/FakeNewsJudgement/vis_log
checkpoint_dir=${method}_${model}_${optimizer}
if [ ${aug} == 1 ]; then
  checkpoint_dir="${checkpoint_dir}_aug"
fi
checkpoint_dir=${checkpoint_dir}_model-${model_id}
tag=${checkpoint_dir}

export CUDA_VISIBLE_DEVICES=${gpu}

echo -----back up methods/${method}.py-----
now=$(date +"%Y%m%d_%T")
cp methods/${method}.py methods/bak/${method}_${model_id}_${now}.py

cmd="
    python run.py
    --mode ${mode}
    --model ${model}
    --optimizer ${optimizer}
    --train_lr ${train_lr}
    --save_freq ${save_freq}
    --start_epoch ${start_epoch}
    --stop_epoch ${stop_epoch}
    --vis_log ${vis_log}
    --tag ${tag}
    --checkpoint_dir ${checkpoint_dir}
    "

if [ ${resume} == 1 ]; then cmd="${cmd} --resume"; fi
if [ ${aug} == 1 ]; then cmd="${cmd} --train_aug"; fi

nohup $cmd >> outputs/nohup_${tag}.output &
train_pid=$!
echo -----cmd------
echo ${cmd}
echo ----output----
echo "tail -f outputs/nohup_${tag}.output"

echo -----tag------
echo Start task of ${tag}...
echo $cmd >> records/${tag}_task.record
echo gpu=${gpu} >> records/${tag}_task.record
echo train_pid=${train_pid} >> records/${tag}_task.record
cat records/${tag}_task.record
echo -------------- >> records/${tag}_task.record