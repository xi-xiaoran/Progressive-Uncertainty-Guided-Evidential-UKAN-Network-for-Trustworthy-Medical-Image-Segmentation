# Progressive-Uncertainty-Guided-Evidential-UKAN-Network-for-Trustworthy-Medical-Image-Segmentation
Link to the paper：[Paper address]()

## Project Description:

Medical image segmentation plays a crucial role in clinical diagnosis and treatment planning, and imposes strict requirements on the accuracy and reliability of models. Most existing methods adopt a strategy based on evidence uncertainty perception. Although it can provide segmentation results and uncertainty measures, achieving a certain degree of trustworthy segmentation, there are still limitations. Firstly, the insufficient learning of difficult samples leads to a significant decrease in the segmentation performance of the model on such samples. Secondly, the low accuracy of evidence uncertainty evaluation limits the model's trustworthy deployment ability. In response to the above issues, this paper proposes a Progressive Uncertainty Guided Empirical UKAN (PUGEUKAN) model for trustworthy medical image segmentation. The model introduces an Uncertainty Attention Block (UAB) module to guide the model to focus on difficult samples using uncertainty maps, thereby improving segmentation accuracy and robustness; Designed an evidence regularization term that integrates prior information and a novel evidence head function to enhance the accuracy of uncertainty estimation; Adopting a multi-scale receptive field module (Multi Dilation Block, MDB) combined with gating mechanism to effectively model the scale differences of lesion areas and further improve segmentation performance. The experimental results on the CVC LinicDB dataset show that PUGEUKAN has improved uncertainty estimation ability by 17.8%, Dice coefficient by 0.6%, and IoU by 0.7% compared to existing mainstream methods such as UKAN and SwinUNETR, significantly verifying its advantages in medical image segmentation. 

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
