#!/bin/bash


# ========== 阶段1：生成解释缓存 (每个模型×方法只跑一次) ==========

# Reveal
for method in gnnexplainer cfexplainer pgexplainer subgraphx deeplift gradcam
do
    python main.py --do_test --do_explain --gnn_model Reveal --ipt_method $method --cuda_id 0
done

# Devign
for method in gnnexplainer cfexplainer pgexplainer subgraphx deeplift gradcam
do
    python main.py --do_test --do_explain --gnn_model Devign --ipt_method $method --cuda_id 0
done

# DeepWukong
for method in gnnexplainer cfexplainer pgexplainer subgraphx deeplift gradcam
do
    python main.py --do_test --do_explain --gnn_model DeepWukong --ipt_method $method --cuda_id 0
done

# IVDetect
for method in gnnexplainer cfexplainer pgexplainer subgraphx deeplift gradcam
do
    python main.py --do_test --do_explain --gnn_model IVDetect --ipt_method $method --cuda_id 0
done


# ========== 阶段2：从缓存评估所有 KM (秒级/次) ==========

# Reveal
for method in gnnexplainer cfexplainer pgexplainer subgraphx deeplift gradcam
do
    for KM in {2..20..2}
    do
        python main.py --do_test --eval_only --gnn_model Reveal --ipt_method $method --KM $KM --cuda_id 0
    done
done

# Devign
for method in gnnexplainer cfexplainer pgexplainer subgraphx deeplift gradcam
do
    for KM in {2..20..2}
    do
        python main.py --do_test --eval_only --gnn_model Devign --ipt_method $method --KM $KM --cuda_id 0
    done
done

# DeepWukong
for method in gnnexplainer cfexplainer pgexplainer subgraphx deeplift gradcam
do
    for KM in {2..20..2}
    do
        python main.py --do_test --eval_only --gnn_model DeepWukong --ipt_method $method --KM $KM --cuda_id 0
    done
done

# IVDetect
for method in gnnexplainer cfexplainer pgexplainer subgraphx deeplift gradcam
do
    for KM in {2..20..2}
    do
        python main.py --do_test --eval_only --gnn_model IVDetect --ipt_method $method --KM $KM --cuda_id 0
    done
done
