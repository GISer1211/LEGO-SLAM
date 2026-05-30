#!/bin/bash

# ---------------------------------------------------------------------------
# SGCO (Semantic-Geometry Co-Optimization) experiment launcher for Replica.
#
# This is a thin wrapper around the SAME pipeline as run_replica.sh, but with
# the SGCO improvements enabled. Keep run_replica.sh as the *baseline* and use
# this script for the "full method" run so you can A/B compare directly.
#
# Usage: bash run_replica_sgco.sh /path/to/Replica
#
# Ablation matrix (override SGCO_FLAGS below):
#   Baseline (= run_replica.sh):  SGCO_FLAGS=""
#   + Innovation 1 (dir loss):    --lambda_feat_cos 0.5 --feat_conf_weight
#   + Innovation 2 (densify):     --sem_densify --sem_densify_grad_th 0.0002 \
#                                 --sem_densify_interval 200 --sem_densify_from_iter 500
#   + Innovation 1b (polarize):   --lambda_sem_contrast 0.05 --sem_contrast_interval 5
# ---------------------------------------------------------------------------

if [ -z "$1" ]; then
    echo "Usage: bash run_replica_sgco.sh <dataset_path>"
    echo "Example: bash run_replica_sgco.sh /path/to/Replica"
    exit 1
fi

OUTPUT_PATH="experiments_sgco"
DATASET_PATH="$1"

# ----- SGCO configuration (the "full method" recommended defaults) -----------
# Innovation 1  : direction-aware + confidence-weighted feature supervision (-> IoU)
# Innovation 2  : semantic-error-guided adaptive densification               (-> PSNR + IoU)
# Innovation 1b : language-field polarization (default OFF; turn on for ablation)
SGCO_FLAGS="--lambda_feat_cos 0.5 --feat_conf_weight \
            --sem_densify --sem_densify_grad_th 0.0002 \
            --sem_densify_interval 200 --sem_densify_from_iter 500 \
            --max_total_gaussians 0"

run_()
{
    local dataset=$1
    local config=$2
    local result_txt=$3
    local keyframe_th=$4
    local knn_maxd=$5
    local overlapped_th=$6
    local max_correspondence_distance=$7
    local trackable_opacity_th=$8
    local overlapped_th2=$9
    local downsample_rate=${10}
    local post_training_iter=${11}
    local eval_ratio=${12}
    local edge_weight=${13}
    local n_trackable_keyframes=${14}
    local pose_lr_rate=${15}
    local loopclosing_global_correspondence_distance=${16}
    local loopclosing_local_correspondence_distance=${17}
    local loop_constraint_noise=${18}
    local loop_closing_start=${19}

    echo "run $dataset (SGCO)"
    echo "" >> ${result_txt}
    echo "run $dataset (SGCO)" >> ${result_txt}
    python -W ignore lego_slam.py --dataset_path $DATASET_PATH/$dataset\
                                    --config $config\
                                    --output_path $OUTPUT_PATH/$dataset/init/\
                                    --keyframe_th $keyframe_th\
                                    --knn_maxd $knn_maxd\
                                    --overlapped_th $overlapped_th\
                                    --max_correspondence_distance $max_correspondence_distance\
                                    --trackable_opacity_th $trackable_opacity_th\
                                    --overlapped_th2 $overlapped_th2\
                                    --downsample_rate $downsample_rate\
                                    --post_training_iter $post_training_iter\
                                    --eval_ratio $eval_ratio\
                                    --save_results \
                                    --speedup \
                                    --system_fps_limit 15.0 \
                                    --n_trackable_keyframes $n_trackable_keyframes\
                                    --enable_loop_closing \
                                    --loop_constraint_noise $loop_constraint_noise \
                                    --loopclosing_global_correspondence_distance $loopclosing_global_correspondence_distance \
                                    --loopclosing_local_correspondence_distance $loopclosing_local_correspondence_distance \
                                    --pose_lr_rate $pose_lr_rate\
                                    --loop_closing_start $loop_closing_start \
                                    --edge_weight ${edge_weight} \
                                    --semantic_feature_init \
                                    --pretrained_encoder_path "saved/cnn_encoder_best.pth" \
                                    --pretrained_decoder_path "saved/cnn_decoder_best.pth" \
                                    ${SGCO_FLAGS} >> ${result_txt}
    wait
}

run_replica()
{
    local result_txt=$1
    local keyframe_th=$2
    local knn_maxd=$3
    local overlapped_th=$4
    local max_correspondence_distance=$5
    local trackable_opacity_th=$6
    local overlapped_th2=$7
    local downsample_rate=$8
    local post_training_iter=$9
    local eval_ratio=${10}
    local edge_weight=${11}
    local n_trackable_keyframes=${12}
    local pose_lr_rate=${13}
    local loopclosing_global_correspondence_distance=${14}
    local loopclosing_local_correspondence_distance=${15}
    local loop_constraint_noise=${16}
    local loop_closing_start=${17}

    for scene in room0 room1 room2 office0 office1 office2 office3 office4; do
        run_ "$scene" "configs/Replica/caminfo.txt" "$result_txt" "$keyframe_th" "$knn_maxd" "$overlapped_th" "$max_correspondence_distance" "$trackable_opacity_th" "$overlapped_th2" "$downsample_rate" "$post_training_iter" "$eval_ratio" "$edge_weight" "$n_trackable_keyframes" "$pose_lr_rate" "$loopclosing_global_correspondence_distance" "$loopclosing_local_correspondence_distance" "$loop_constraint_noise" "$loop_closing_start"
    done
}

txt_file="replica_results_sgco.txt"

overlapped_th=1e-3
max_correspondence_distance=0.02
knn_maxd=99999.0
trackable_opacity_th=0.05
overlapped_th2=1e-4
downsample_rate=10
keyframe_th=0.75
post_training_iter=0
eval_ratio=1.0
edge_weight=0.1
n_trackable_keyframes=100
pose_lr_rate=0.1
loopclosing_global_correspondence_distance=0.1
loopclosing_local_correspondence_distance=0.03
loop_constraint_noise=1e-2
loop_closing_start=10

run_replica $txt_file $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance \
      $trackable_opacity_th $overlapped_th2 $downsample_rate $post_training_iter $eval_ratio $edge_weight $n_trackable_keyframes $pose_lr_rate $loopclosing_global_correspondence_distance $loopclosing_local_correspondence_distance $loop_constraint_noise $loop_closing_start
