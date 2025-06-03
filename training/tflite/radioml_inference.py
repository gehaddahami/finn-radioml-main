import os
import pathlib
import time
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image
import numpy as np

# Specify the TensorFlow model, labels, and image
script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, 'model_quant_edgetpu.tflite')
#label_file = os.path.join(script_dir, 'imagenet_labels.txt')
#image_file = os.path.join(script_dir, 'parrot.jpg')

# Initialize the TF interpreter
interpreter = edgetpu.make_interpreter(model_file)

batch_size = 1024
interpreter.resize_tensor_input(0, [batch_size, 1, 1024, 2])
interpreter.allocate_tensors()

# Resize the image
size = common.input_size(interpreter)
#image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)

# Run an inference
#common.set_input(interpreter, image)
#interpreter.invoke()
#classes = classify.get_classes(interpreter, top_k=1)

# generate flattened input tensor (batch 1)
input_data = np.random.randint(-127, 128, size=(batch_size, 1, 1024, 2), dtype=np.int8)
#input_data = np.random.randint(-127, 128, size=(batch_size*1*1024*2), dtype=np.int8)

print("Input data:")
print(input_data)

interpreter.set_tensor(0, input_data)

# Run inference
print('----INFERENCE TIME----')
print('Note: The first inference on Edge TPU is slow because it includes',
    'loading the model into Edge TPU memory.')
inference_times = []
for _ in range(1000):
    start = time.perf_counter()
    interpreter.invoke()
    #edgetpu.run_inference(interpreter, input_data)
    inference_time = time.perf_counter() - start
    #output_data = common.output_tensor(interpreter, 0)
    #print("Output data:")
    #print(output_data.shape)
    #print(output_data)
    print('%.3fms' % (inference_time * 1000))
    inference_times.append(inference_time)

inference_times = inference_times[100:] #throw away first 100 runs (warmup)

print('Minimum: %.3fms' % (min(inference_times) * 1000))
print('Average: %.3fms' % (np.mean(inference_times) * 1000))
# Print the result
#labels = dataset.read_label_file(label_file)
#for c in classes:
#  print('%s: %.5f' % (labels.get(c.id, c.id), c.score))
