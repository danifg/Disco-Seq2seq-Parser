


for split in dev test train
do


python ./linearization/scripts/get_inorder_oracle_SWAP.py disco_data/train.discbracket disco_data/$split.discbracket > DATA/dep-parsing/corpora/${1}/$split.actions


python ./linearization/scripts/get_words.py en disco_data/train.discbracket disco_data/$split.discbracket > DATA/dep-parsing/corpora/${1}/$split.en

python ./linearization/scripts/get_words_with_index.py en disco_data/train.discbracket disco_data/$split.discbracket > DATA/dep-parsing/corpora/${1}/$split.source

python ./linearization/scripts/get_tags.py en disco_data/train.discbracket disco_data/$split.discbracket > DATA/dep-parsing/corpora/${1}/$split.pos


cp disco_data/$split.discbracket DATA/dep-parsing/corpora/${1}/$split.discbracket


done
