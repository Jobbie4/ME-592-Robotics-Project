# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 22:37:52 2025

@author: markg
"""

# === train_model.py ===
import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Lambda
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


IMAGE_HEIGHT, IMAGE_WIDTH = 66, 200
BATCH_SIZE = 32
EPOCHS = 30

# --------- Utilities ---------
def build_model():
    inputs = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))

    x = Lambda(lambda x: x)(inputs)
    x = Conv2D(24, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Conv2D(36, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Conv2D(48, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    x = Dense(10, activation='relu')(x)

    steer_output = Dense(1, name='steering_output')(x)
    throttle_output = Dense(1, name='throttle_output')(x)

    return Model(inputs=inputs, outputs=[steer_output, throttle_output])

def load_dataset(csv_path, image_folder):
    df = pd.read_csv(csv_path, delimiter=';')
    df = df.dropna(subset=['Path', 'SteerAngle', 'Throttle'])
    df['ImageFile'] = df['Path'].apply(lambda p: os.path.basename(p))

    def load_image(filename):
        full_path = os.path.join(image_folder, filename)
        image = cv2.imread(full_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        return image / 255.0

    images = np.array([load_image(fname) for fname in df['ImageFile']])
    steering = df['SteerAngle'].astype(np.float32).values
    throttle = df['Throttle'].astype(np.float32).values

    return images, steering, throttle

# --------- Training ---------
train_csv_path = r"C:\Mark's Python files\ME5920\HW3_Gardocki\DefinedTurns\robot_log.csv"
train_image_folder = r"C:\Mark's Python files\ME5920\HW3_Gardocki\DefinedTurns\IMG"

X_train, y_steer_train, y_throttle_train = load_dataset(train_csv_path, train_image_folder)

model = build_model()
model.compile(optimizer='adam',
              loss={'steering_output': 'mse', 'throttle_output': 'mse'},
              metrics={'steering_output': 'mse', 'throttle_output': 'mse'})

history = model.fit(
    X_train,
    {'steering_output': y_steer_train, 'throttle_output': y_throttle_train},
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True
)


plt.plot(history.history['steering_output_loss'], label='Steer Loss (Train)')
plt.plot(history.history['val_steering_output_loss'], label='Steer Loss (Val)')
plt.plot(history.history['throttle_output_loss'], label='Throttle Loss (Train)')
plt.plot(history.history['val_throttle_output_loss'], label='Throttle Loss (Val)')
plt.title('Training Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid()
plt.show()

model.save("dual_output_model_3.keras")
print("Model saved to disk.")


#%%
# === test_model.py ===
import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

IMAGE_HEIGHT, IMAGE_WIDTH = 66, 200

def load_dataset(csv_path, image_folder):
    df = pd.read_csv(csv_path, delimiter=';')
    df = df.dropna(subset=['Path', 'SteerAngle', 'Throttle'])
    df['ImageFile'] = df['Path'].apply(lambda p: os.path.basename(p))

    def load_image(filename):
        full_path = os.path.join(image_folder, filename)
        image = cv2.imread(full_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        return image / 255.0

    images = np.array([load_image(fname) for fname in df['ImageFile']])
    steering = df['SteerAngle'].astype(np.float32).values
    throttle = df['Throttle'].astype(np.float32).values

    return images, steering, throttle

# --------- Testing ---------
test_csv_path = r"C:\Users\markg\Downloads\test_3\test_3\robot_log.csv"
test_image_folder = r"C:\Users\markg\Downloads\test_3\test_3\IMG"

model = load_model("dual_output_model_3.keras", safe_mode=False)
X_test, y_steer_test, y_throttle_test = load_dataset(test_csv_path, test_image_folder)

pred_steer, pred_throttle = model.predict(X_test)
losses = model.evaluate(X_test, {
    'steering_output': y_steer_test,
    'throttle_output': y_throttle_test
})
print("Evaluation:", losses)

# Optional visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_steer_test, pred_steer, alpha=0.5)
plt.plot([min(y_steer_test), max(y_steer_test)],
         [min(y_steer_test), max(y_steer_test)], 'r--')
plt.title('Steering: Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.subplot(1, 2, 2)
plt.scatter(y_throttle_test, pred_throttle, alpha=0.5)
plt.plot([min(y_throttle_test), max(y_throttle_test)],
         [min(y_throttle_test), max(y_throttle_test)], 'r--')
plt.title('Throttle: Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.tight_layout()
plt.show()
