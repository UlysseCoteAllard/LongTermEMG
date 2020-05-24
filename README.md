# LongTermEMG

Work in progress. The repository should change to be more user friendly.

LongTerm 3DC Dataset is available here: http://ieee-dataport.org/1948

3DC Dataset is available here: https://github.com/UlysseCoteAllard/sEMG_handCraftedVsLearnedFeatures

First prepare the longterm 3DC Dataset by running: PrepareAndLoadDataLongTerm->prepare_from_raw_dataset.py Then in TrainingsAndEvaluations all the files are there which were used to obtain the results from: Virtual Reality to Study the Gap Between Offline and Real-Time EMG-based Gesture Recognition https://arxiv.org/abs/1912.09380

And

Unsupervised Domain Adversarial Self-Calibration for Electromyographic-based Gesture Recognition

In TrainingsAndEvaluations->ForTrainingSessions you have the different mains to train the algorithms (spectrograms refers to the Unsupervised Domain Adversarial paper, otherwise it's the Virtual Reality paper). In the TrainingsAndEvaluations->self_learning you have the files to train both SCADANN and MV. Then TrainingsAndEvaluations->ForEvaluationSessions are the files using the evaluation sessions for the Virtual Reality paper. TrainingsAndEvaluations->SpectrogramEvaluationSessions are the files using the evaluation sessions for the Unsupervised Domain Adversarial paper.

#Required libraries:

The VADA and Dirt-T implementation is based on: https://github.com/ozanciga/dirt-t and https://github.com/RuiShu/dirt-t

Numpy https://numpy.org/

SciPy https://www.scipy.org/

Scikit-learn http://scikit-learn.org/stable/

Sampen https://pypi.org/project/sampen/

PyWavelets https://pywavelets.readthedocs.io/en/latest/

Matplotlib https://matplotlib.org/

Pytorch https://pytorch.org/
