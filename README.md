**ME-5920 Final Project – Human Motion Prediction with LSTMs**
**Team Members:**

*Eric Upchurch*

*Mark Gardoki*

*Christopher Santillan*


**Repository Structure**

**Final_project/**

**Top-Level Scripts**

convertData.py

↳ *Converts raw dataset into frame_id, person_id, x, y format.*

Hyper_lstmGridExperiments.py

↳ *Trains a grid of LSTM models using multiple hyperparameter combinations.*

Hyper_lstmModelAnalysis.py

↳ *Generates graphs to compare trained models.
-->Best run inside a model folder (e.g., Model_2_config_25_vary_pred_seq_len).*

realTimeGraph.py

↳ *Visualizes predicted vs actual trajectories in real time.
--> Tip: You can test this by changing exp_id, the .json config, and dataset files.*

lstm_model_6/ and lstm_model_8/

↳ *Model 6 & 8 evaluation results.*

![Model 6](https://github.com/user-attachments/assets/b98ebc57-8441-4a2c-acf2-46c38daee175)
![Model 8](https://github.com/user-attachments/assets/eb33166e-0743-4a1d-b77b-85e0f8d0b0b9)


**Data Folder**

crowds/data/

↳ *Contains all raw datasets used in the project:*

Zara01/

Zara02/

Arxiwpiskopi_flock/

**Trained Model Batches**

Models_Trained_on_Final_Presentation/
      
↳ *Final models trained on Zara01 for our presentation.*

Models_Trained_with_vary_seq_and_pred_len/

Model_2_config_25_vary_pred_seq_len/

↳ *Trained models and scripts for visual analysis.*

Model_3_config_25_vary_pred_seq_len_epochs_20_30/

↳ *Models trained with longer epochs (20 and 30).*
