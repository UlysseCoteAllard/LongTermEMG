# LongTermEMG

LongTerm 3DC Dataset as presentend in: Côté-Allard, Ulysse, et al. "A Transferable Adaptive Domain Adversarial Neural Network for Virtual Reality Augmented EMG-Based Gesture Recognition." IEEE Transactions on Neural Systems and Rehabilitation Engineering 29 (2021): 546-555.

LongTerm 3DC Dataset is available here: http://ieee-dataport.org/1948

3DC Dataset is available here: https://github.com/UlysseCoteAllard/sEMG_handCraftedVsLearnedFeatures



First prepare the longterm 3DC Dataset by running: PrepareAndLoadDataLongTerm->prepare_from_raw_dataset.py Then in TrainingsAndEvaluations all the files are there which were used to obtain the results from: Côté-Allard, U., Gagnon-Turcotte, G., Phinyomark, A., Glette, K., Scheme, E., Laviolette, F., & Gosselin, B. (2021). A Transferable Adaptive Domain Adversarial Neural Network for Virtual Reality Augmented EMG-Based Gesture Recognition. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 29, 546-555. (https://arxiv.org/abs/1912.09380v2)

And

Côté-Allard, U., Gagnon-Turcotte, G., Phinyomark, A., Glette, K., Scheme, E. J., Laviolette, F., & Gosselin, B. (2020). Unsupervised domain adversarial self-calibration for electromyography-based gesture recognition. IEEE Access, 8, 177941-177955. (https://ieeexplore.ieee.org/document/9207910)

In TrainingsAndEvaluations->ForTrainingSessions you have the different mains to train the algorithms (spectrograms refers to the Unsupervised Domain Adversarial paper, otherwise it's the Virtual Reality paper). In the TrainingsAndEvaluations->self_learning you have the files to train both SCADANN and MV. Then TrainingsAndEvaluations->ForEvaluationSessions are the files using the evaluation sessions for the Virtual Reality paper. TrainingsAndEvaluations->SpectrogramEvaluationSessions are the files using the evaluation sessions for the Unsupervised Domain Adversarial paper.

#Required libraries:

The VADA and Dirt-T implementation are based on: https://github.com/ozanciga/dirt-t and https://github.com/RuiShu/dirt-t

Numpy https://numpy.org/

SciPy https://www.scipy.org/

Scikit-learn http://scikit-learn.org/stable/

Sampen https://pypi.org/project/sampen/

PyWavelets https://pywavelets.readthedocs.io/en/latest/

Matplotlib https://matplotlib.org/

Pytorch https://pytorch.org/
