FocusPal

##Overview 📝

FocusPal is an AI powered tool that uses machine learning to help students maintain focus while studying. FocusPal uses a neural network on a video of the user to determine whether they are currently focused or unfocused on their task. What's special about FocusPal is that it detects focused and unfocused modes while working off their computers. It can clearly differentiate between users being on their phone and reading off a textbook. It also uses clever puns to help students study.

##Build Process

Trained a pre-trained model (ResNet152v2) on relevant data in Google Colab notebooks. The CNN binary classifier was trained on individual frames from the training videos, and was able to obtain high accuracy. This model was called every few frames to analyze the user. If the user is detected as unfocused multiple times in a row, it gives a warning to the user.


##Built with
opencv
ResNet152v2
tensorflow (keras)

##🧑‍💻 Authors 🧑‍💻
Midhuna S / @midhunaS


### Model Download
The trained model is too large for GitHub.  
(Download it here from Google Drive)(https://drive.google.com/drive/folders/1AqZjDCTtLoQd08Y-DRm2EwNIuRPzjIkU?usp=drive_link).
Place it inside `model/variables/`.
