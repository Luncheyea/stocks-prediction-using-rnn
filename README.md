# Stocks Prediction Using RNN

This project was developed as part of the Artificial Neural Network course within the Department of CSIE at National Taiwan Normal University. Its primary objective is to utilize neural networks for the prediction of the 0050.TW stock index.<br>
For more detailed insights, the technical report associated with this project is available [[here](Stocks_Prediction_Using_RNN.pdf)].<br>

## Prerequisites

- Python >= 3.10.14
- pandas >= 2.2.1
- Matplotlib >= 3.8.3
- PyTorch >= 2.2.1

## Training

```shell
python train.py --model "./model(lr=0.0001 epoch=150).pth" \
      --dataset "./dataset/0050.TW close.csv" --lr 0.0001 --epochs 150 
```

## Evaluation
```shell
python eval.py --model "./model(lr=0.0001 epoch=150).pth" \
     --dataset "./dataset/eval.csv"
```

## Model Weight
| Model Weight | learning rate | epochs |
| :----------: | :-----------: | :----: |
| [link](model(lr%3D0.0001%20epoch%3D150).pth) | 0.0001 | 150 |

## License
This project is licensed under [MIT License](LICENSE), as found in the LICENSE file.
