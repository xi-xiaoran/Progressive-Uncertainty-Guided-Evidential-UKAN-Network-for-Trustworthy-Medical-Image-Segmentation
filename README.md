# Progressive-Uncertainty-Guided-Evidential-UKAN-Network-for-Trustworthy-Medical-Image-Segmentation
Link to the paper：[Paper address]()

## Project Description:

We propose PUGEUKAN, which introduces the Uncertainty-Aware Attention Block (UAB) that leverages uncertainty maps to guide the segmentation process, as well as an evidence-based uncertainty progressive guidance method for robust and accurate medical image segmentation. This model enhances the focus on hard samples, thereby improving the segmentation accuracy and robustness. Furthermore, we integrate a novel module called the Multi-scale Dilation Block (MDB), which utilizes a gating mechanism to combine micro-step and dilated convolutions. This allows the model to adaptively adjust its receptive field to extract multi-scale features, boosting its uncertainty awareness and enabling it to handle medical images with varying lesion sizes. Additionally, we design a new evidence head function and introduce an evidence regularization term incorporating prior knowledge to strengthen the model’s uncertainty estimation, ultimately ensuring reliable model deployment.

![model](https://github.com/xi-xiaoran/Progressive-Uncertainty-Guided-Evidential-UKAN-Network-for-Trustworthy-Medical-Image-Segmentation/blob/main/Plot/model.PNG)

![Display image of segmentation effect](https://github.com/xi-xiaoran/Progressive-Uncertainty-Guided-Evidential-UKAN-Network-for-Trustworthy-Medical-Image-Segmentation/blob/main/Plot/result.PNG)
## Installation:

This project requires the following Python libraries:

torch 1.13.1  
torchvision 0.14.1  
numpy 1.21.6  
scikit-learn 1.0.2  

(Anyway, the author is using the version above)
## Usage:
The `model folder` contains the main model architecture.  
The `Loss_function folder` contains the loss functions for evidence learning.  
The `deal_data folder` contains the data loader setup.  

- `How to draw beautiful pictures.pptx` file contains instructions for drawing the model architecture, which can be used as a reference.

- `main.py` includes the main settings for evidence learning.

- `train_model.py` contains the training code for the evidence learning model.

- `Test_model.py` contains the evaluation code for the evidence learning model.

- `Bys.py` includes the main settings for the Bayesian model.

- `Bys_train.py` contains the training code for the Bayesian model.

- `Bys_test.py` contains the evaluation code for the Bayesian model.

- `Calibration.py` contains the main settings for the calibration model.

- `Calibration_train.py` contains the training code for the calibration model.

- `Calibration_test.py` contains the evaluation code for the calibration model.

- `Plot.py` contains functions for visualizing the results, plotting predictions and uncertainty maps of multiple models, and saving them in the show folder.

- `Plotandsave.py` contains plotting functions that save the predictions, uncertainty maps, and prediction errors of multiple models in the show folder.

- `jpg_to-png.py` contains functions to convert datasets from jpg format to png format

- `move.py` contains a function that can separate the test set from the dataset in a txt file

- `ASSD.py` Contains functions for calculating ASSD values

## Contact the Author:

You can contact the author via email at 2646594598@qq.com.
(The author may be slacking off and not responding to messages)
