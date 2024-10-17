
# a gradio demo for comparing DDIM, IADB/RectifiedFlow and BNDM on the church 64x64 dataset

python gradio_bndm.py --dataset=church_res64 --res=64 --train_or_test=test --scheduler_gamma=sigmoid --scheduler_param=1000 --nb_steps=50

