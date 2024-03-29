import tensorflow as tf
#from tensorflow.contrib.lite.python import convert_saved_model
#saved_model_dir = './blue_point_model/'
#converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
#tflite_model = converter.convert()
#open("converted_model.tflite", "wb").write(tflite_model)

# Construct a basic model.
root = tf.train.Checkpoint()
root.v1 = tf.Variable(3.)
root.v2 = tf.Variable(2.)
root.f = tf.function(lambda x: root.v1 * root.v2 * x)

# Save the model in SavedModel format.
export_dir = "./blue_point_model/"
input_data = tf.constant(1., shape=[1, 300, 300, 3])
to_save = root.f.get_concrete_function(input_data)
tf.saved_model.save(root, export_dir, to_save)


#model = tf.saved_model.load(export_dir)
#concrete_func = model.signatures[
#tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
#concrete_func.inputs[0].set_shape([1, 300, 300, 3])
#converter = TFLiteConverter.from_concrete_functions([concrete_func])



# Convert the model.
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
tflite_model = converter.convert()

# Save the TF Lite model.
with tf.io.gfile.GFile(export_dir + 'model.tflite', 'wb') as f:
  f.write(tflite_model)

