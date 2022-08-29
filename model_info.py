import tensorflow as tf

def standard_model(model_path):
    model = tf.saved_model.load(model_path+"tf_model")
    print(model.summary())

def quantized_model(model_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path+"tf_model")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    tf.lite.experimental.Analyzer.analyze(model_content=tflite_quant_model)

def main(argument):
    #arguments
    model_path = "output/"
    quantized_model(model_path)


if __name__ == "__main__":
    argument = "c"
    main(argument)



