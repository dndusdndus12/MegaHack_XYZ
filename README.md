# MegaHack_XYZ
Challenge Smart Tech - 4 Malware Hunt


## Model Training
Dataset – 8 types of malwares [([1](https://github.com/Endermanch/MalwareDatabase)),([2](https://github.com/cryptwareapps/Malware-Database))]

No of files: 100 +

Each malware image is fragmented

The batch size is set to 64, and training is conducted for 5-10 epochs.

We reduced the number of epochs due to limited time and resources available for training

## Model Evaluation:
The model is evaluated using the test data (20% randomly chosen).

Testing Accuracy > 80% (In Literature > 95% [3])


## References
[1] https://github.com/Endermanch/MalwareDatabase

[2] https://github.com/cryptwareapps/Malware-Database 

[3] S. Tobiyama, Y. Yamaguchi, H. Shimada, T. Ikuse and T. Yagi, "Malware Detection with Deep Neural Network Using Process Behavior," 2016 IEEE 40th Annual Computer Software and Applications Conference (COMPSAC), Atlanta, GA, USA, 2016, pp. 577-582, doi: 10.1109/COMPSAC.2016.151.
