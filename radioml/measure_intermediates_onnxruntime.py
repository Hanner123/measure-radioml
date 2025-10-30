# ir.Model, ir.from_proto, ir.to_proto, ...
import onnx_ir as ir
import numpy as np
# The runtime simply builds a wrapper around ONNX Runtime for model execution
import onnxruntime
import onnx
model_path = "/home/hanna/git/measure-radioml/inputs/radioml/model_brevitas_1_simpl.onnx"
data_path = "/home/hanna/git/radioml-transformer/data/GOLD_XYZ_OSC.0001_1024.npz"
# 1. ONNX-Modell laden (Protobuf-Format)
onnx_model = onnx.load(model_path)

# 2. ONNX-Modell in IR-Modell umwandeln
model = ir.from_proto(onnx_model)


# Make a deep copy of the model to not mess up the graph by executing it...
model = ir.from_proto(ir.to_proto(model))
# Remember the original list of outputs before extending by all
# intermediate tensors
outputs = list(model.graph.outputs)

# Start collecting a list of intermediate tensor value information
intermediates = []

# Optionally extends the list of model outputs by all intermediate tensors,
# i.e., produces a full execution context dump
full_context_dump = True
if full_context_dump:
    # Collect all tensors which are actually used in the graph, i.e.,
    # connected to any node as either input or output
    for node in ir.traversal.RecursiveGraphIterator(model.graph):
        intermediates.extend([*node.inputs, *node.outputs])

# Filter out all tensors which are already graph outputs and remove all
# duplicates by turning the list into a set
intermediates = {x for x in intermediates if not x.is_graph_output()}
# Extend the list of graph outputs by all these intermediate (or input)
# tensors - this keeps the original outputs first
model.graph.outputs.extend(intermediates)

# Convert the model to a string-serialized protobuf representation
# understood by ONNX Runtime
model = ir.to_proto(model).SerializeToString()  #ansonsten als onnx speichern und das modell in measure laden

# Disable further ONNX Runtime session graph optimizations
sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = (
    onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
)
# Only show error and fatal messages
sess_options.log_severity_level = 3

# Create an inference session from the ONNX model converted to proto
# representation
session = onnxruntime.InferenceSession(model, sess_options, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
# Laden der Daten
data = np.load(data_path)
X = data["X"]  # Shape: (samples, 1024, 2)
Y = data["Y"]  # Shape: (samples,)

# pred label 4
# true label 6
x_sample = X[4684]            # (1024, 2)
y_label = Y[4684]             # int Label



# Eingabe für das Modell: (1, 1, 1024, 2)
x_input = np.expand_dims(x_sample, axis=0)  # (1, 1024, 2)


context = {input_name: x_input.astype(np.float32)}
# Evaluate the model on the inputs form the execution context by running the
# prepared inference session and collect all outputs as results
results = session.run(None, context)
# Collect full execution context by associating each output, including the
# intermediates, to its name
context = {
    **{out.name: x for x, out in zip(results[:len(outputs)], outputs)},
    **{out.name: x for x, out in zip(results[len(outputs):], intermediates)}
}

# print(context)
np.savez("context_output_jetson.npz", **context)

outputs = [x for x, _ in zip(results, outputs)]
print(outputs)

pred_label = np.argmax(outputs[0], axis=1)[0]

print("pred label:", pred_label, "true label:", y_label)