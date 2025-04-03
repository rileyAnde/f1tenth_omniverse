import numpy as np
import tensorflow as tf
import time

# load model once at startup
class Callable_TLN():
    def __init__(self):
        MODEL_PATH = "/home/r478a194/Downloads/f1_tenth_model_small_noquantized.tflite"
        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()

        #get input/output tensor indices
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]

    def linear_map(self, x, x_min, x_max, y_min, y_max):
        return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min    

    def tln_expert(self, observation):

        scans = np.array(observation)
        scans = np.append(scans, [20])

        # add noise and normalize
        noise = np.random.normal(0, 0.5, scans.shape)
        scans = np.clip(scans + noise, 0, 10)  # Clip max range to 10m
        scans = scans[:541]  # Downsample to reduce input size
        scans = np.expand_dims(scans, axis=-1).astype(np.float32)
        scans = np.expand_dims(scans, axis=0)  # Add batch dimension

        # run inference
        start_time = time.time()
        self.interpreter.set_tensor(self.input_index, scans)
        self.interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Get model output
        output = self.interpreter.get_tensor(self.output_index)
        steer = output[0, 0]
        speed = output[0, 1]

        # Scale speed to match real-world limits
        min_speed, max_speed = 0, 1
        speed = self.linear_map(speed, 0, 1, min_speed, max_speed)

        return np.array([steer, speed])  # return as NumPy array
