

# fig 6: lsun_church (32->128), superres, iadb (gaussian) and ours (gaussianBN)

python iadb_bn.py --dataset=church_res128 --res=128 --batch_size=200 --train_or_test=test --nb_steps=250 --test_samples=100 --is_conditional --noise_type=gaussian --scheduler_gamma=linear --scheduler_param=1 --out_channel=3 --conditional_type=superres

python iadb_bn.py --dataset=church_res128 --res=128 --batch_size=200 --train_or_test=test --nb_steps=250 --test_samples=100 --is_conditional --noise_type=gaussianBN --scheduler_gamma=sigmoid --scheduler_param=0.2 --out_channel=6 --conditional_type=superres

