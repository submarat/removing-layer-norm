apt install git-lfs
git lfs install
git clone https://huggingface.co/datasets/NeelNanda/pile-10k
mv pile-10k/data/train-00000-of-00001-4746b8785c874cc7.parquet raw_pile10k.parquet
rm -r pile-10k
