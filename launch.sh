export RUN="docker run --gpus device=${CUDA_VISIBLE_DEVICES} -e PATH=/cig/anaconda/bin -e TRANSFORMERS_CACHE=/cig/transformers -u $(id -u):$(id -g) -v $(dirname $(realpath ${0})):/cig -w /cig -it nvidia/cuda:11.6.0-base-ubuntu18.04"

if [ ${1} = matching ]; then
  ${RUN} python main.py \
    --stage ${1} \
    --matching_loss_scale 1.0 \
    --diffusion_loss_scale 0.0 \
    --noisy_matching_loss_scale 0.0

elif [ ${1} = diffusion ]; then
  ${RUN} python main.py \
    --stage ${1} \
    --matching_loss_scale 0.1 \
    --diffusion_loss_scale 1.0 \
    --noisy_matching_loss_scale 0.0

elif [ ${1} = noisy_matching ]; then
  ${RUN} python main.py \
    --stage ${1} \
    --matching_loss_scale 0.0 \
    --diffusion_loss_scale 0.0 \
    --noisy_matching_loss_scale 1.0

elif [ ${1} = inference ]; then
  ${RUN} python main.py --stage ${1}
  ${RUN} python execute.py

else
  exit 1
fi
