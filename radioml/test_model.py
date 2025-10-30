import numpy as np
import onnxruntime as ort
import torch


import sys
print("onnxruntime:", ort.__version__)
print("available providers:", ort.get_available_providers())


model_path = "/home/hanna/git/measure-radioml/inputs/radioml/model_dynamic_batchsize.onnx"
model = "/home/hanna/git/measure-radioml/inputs/radioml/model_brevitas_1_simpl.onnx"
#model_path = "/home/hanna/git/measure-radioml/inputs/radioml/model_brevitas_1.onnx"
data_path = "/home/hanna/git/radioml-transformer/data/GOLD_XYZ_OSC.0001_1024.npz"


# sess_options = ort.SessionOptions()
# sess_options.intra_op_num_threads = 1
# session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
# input_name = session.get_inputs()[0].name
# output_name = session.get_outputs()[0].name

sess_options = ort.SessionOptions()
# Das ist sehr wichtig!!!
sess_options.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_DISABLE_ALL
)
sess_options.log_severity_level = 3
session = ort.InferenceSession(model, sess_options, providers=['CPUExecutionProvider'])

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

data = np.load(data_path)
X = data["X"] 
Y = data["Y"]  

num_samples = 20

np.random.seed(42)
indices = np.random.choice(len(X), size=num_samples, replace=False)
correct = 0
total = 0
for idx in indices:
    x_sample = X[idx]            
    y_label = Y[idx]             

    
    x_input = np.expand_dims(x_sample, axis=0)  # (1, 1, 1024, 2)

    # ONNX Inferenz (numpy float32 Input)
    pred_onnx = session.run([output_name], {input_name: x_input.astype(np.float32)})[0]  # (1, 24)

    # Prediction Label (argmax)
    pred_label = np.argmax(pred_onnx, axis=1)[0]
    print(pred_onnx)

    print("pred label:", pred_label, "true label:", y_label)

    # print(f"Prediction: [{pred_label}]  Ground Truth: [{y_label}]")
    if pred_label == y_label:
        correct = correct + 1
    total = total + 1
print("Accuracy: ", correct / total)



    # modell und testdaten sollten auf jetson und pC die gleichen sein (von PC auf jetson kopiert)
    # testen ob es die gleichen sind: mit hash prüfen -> sind gleich

    # 60% bei int 8 mit onnxruntime und PC
    # 37% bei int 8 mit onnxruntime und jetson
    # 4% bei int 8 mit tensorrt und jetson
 