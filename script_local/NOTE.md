# Note

## parameter settings

Mitsuhara's settings from README and arg_parser is as follows:

```bash
python3 main.py -a resnet101 --data CelebA
    --epochs 10
    --schedule 5 7 (--gamma 0.1)
    --train_batch 32
    --lr 0.01
    --momentum 0.9
    --weight_decay 1e-4
```

The main problem is the number of trainig epochs and decreasing learning rate.
