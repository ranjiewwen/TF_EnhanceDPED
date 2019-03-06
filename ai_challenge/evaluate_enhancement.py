from __future__ import print_function

from ai_challenge.ssim import MultiScaleSSIM

import tensorflow as tf
from scipy import misc
import numpy as np
import utils
import os


## --------- Change test parameters below -----------

# from models import srcnn as test_model                  # import your model definition as "test_model"
# model_location = "models_pretrained/dped_srcnn"         # specify the location of your saved pre-trained model (ckpt file)

# from exp4_8_01_unet import unet as test_model
# model_location = "models_pretrained/iphone_iteration_55000.ckpt"

from ai_challenge.models import resnet_12_64 as test_model
model_location = "ai_challenge/models_pretrained/dped_resnet_12_64"

compute_PSNR_SSIM = True
compute_running_time = True
compute_Image = False

if __name__ == "__main__":

    print("\n-------------------------------------\n")
    print("Image Image Enhancement task\n")

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    np.warnings.filterwarnings('ignore')

    ###############################################################
    #  1 Produce .pb model file that will be used for validation  #
    ###############################################################

    print("Saving pre-trained model as .pb file")

    g = tf.Graph()
    with g.as_default(), tf.Session() as sess:

        image_ = tf.placeholder(tf.float32, shape=(1, None, None, 3), name="input")
        out_ = tf.identity(test_model(image_), name="output")

        saver = tf.train.Saver()
        saver.restore(sess, model_location)

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, g.as_graph_def(), "input,output".split(",")
        )

        tf.train.write_graph(output_graph_def, 'ai_challenge/models_converted', 'model_final.pb', as_text=False)

    print("Model was successfully saved!")
    print("\n-------------------------------------\n")
    sess.close()


    if compute_PSNR_SSIM:

        #######################################
        #  2 Computing PSNR / MS-SSIM scores  #
        #######################################

        tf.reset_default_graph()
        config = None

        with tf.Session(config=config) as sess:

            print("\rLoading pre-trained model")

            with tf.gfile.FastGFile("ai_challenge/models_converted/model_final.pb", 'rb') as f:

                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
                print(graph_def)

                x_ = sess.graph.get_tensor_by_name('input:0')
                out_ = sess.graph.get_tensor_by_name('output:0')

            y_ = tf.placeholder(tf.float32, [1, None, None, 3])

            output_crop_ = tf.clip_by_value(out_, 0.0, 1.0)
            target_crop_ = tf.clip_by_value(y_, 0.0, 1.0)

            psnr_ = tf.image.psnr(output_crop_, target_crop_, max_val=1.0)

            print("Computing PSNR/SSIM scores....")

            ssim_score = 0.0
            psnr_score = 0.0
            validation_images = os.listdir("ai_challenge/dped/patches/canon/")
            num_val_images = len(validation_images)

            for j in range(num_val_images):

                image_phone = misc.imread("ai_challenge/dped/patches/iphone/" + validation_images[j])
                image_dslr = misc.imread("ai_challenge/dped/patches/canon/" + validation_images[j])

                image_phone = np.reshape(image_phone, [1, image_phone.shape[0], image_phone.shape[1], 3]) / 255
                image_dslr = np.reshape(image_dslr, [1, image_dslr.shape[0], image_dslr.shape[1], 3]) / 255

                [psnr, enhanced] = sess.run([psnr_, out_], feed_dict={x_: image_phone, y_: image_dslr})

                psnr_score += psnr / num_val_images
                ssim_score += MultiScaleSSIM(image_dslr * 255, enhanced * 255) / num_val_images

            print("\r\r\r")
            print("Scores | PSNR: %.4g, MS-SSIM: %.4g" % (psnr_score, ssim_score))

            print("\n-------------------------------------\n")
            sess.close()


    if compute_running_time:

        ##############################
        #  3 Computing running time  #
        ##############################

        print("Evaluating model speed")
        print("This can take a few minutes\n")

        tf.reset_default_graph()

        print("Testing pre-trained baseline SRCNN model")
        avg_time_baseline, max_ram = utils.compute_running_time("superres", "ai_challenge/models_pretrained/dped_srcnn.pb", "ai_challenge/dped/HD_res/")

        tf.reset_default_graph()

        print("Testing provided model")
        avg_time_solution, max_ram = utils.compute_running_time("superres", "ai_challenge/models_converted/model.pb", "ai_challenge/dped/HD_res/")

        print("Baseline SRCNN time, ms: ", avg_time_baseline)
        print("Test model time, ms: ", avg_time_solution)
        print("Speedup ratio (baseline, ms / solution, ms): %.4f" % (float(avg_time_baseline) / avg_time_solution))
        print("Approximate RAM consumption (HD image): " + str(max_ram) + " MB")

        scoreA = 4 * (psnr_score - 21) + 100 * (ssim_score - 0.9) + 2 * min(float(avg_time_baseline) / avg_time_solution,4)
        scoreB = 1 * (psnr_score - 21) + 400 * (ssim_score - 0.9) + 2 * min(float(avg_time_baseline) / avg_time_solution,4)
        scoreC = 2 * (psnr_score - 21) + 200 * (ssim_score - 0.9) + 2.9 * min(float(avg_time_baseline) / avg_time_solution,4)

        print("------------------------------\n")
        print("scoreA: %0.4g ; scoreB: %.4g ; scoreC: %.4g ." % (scoreA,scoreB,scoreC) )
        print("------------------------------\n")

    if compute_Image:

        print('Generate Test Image')
        tf.reset_default_graph()
        config = None
        with tf.Session(config=config) as sess:
            with tf.gfile.FastGFile("ai_challenge/models_converted/model.pb", 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')

                x = sess.graph.get_tensor_by_name('input:0')
                out = sess.graph.get_tensor_by_name('output:0')

                output = tf.cast(255.0 * tf.squeeze(tf.clip_by_value(out, 0, 1)), tf.uint8)

            input_path = "ai_challenge/dped/full_size_test_images/"
            test_images = os.listdir(input_path)
            test_images=[elem for elem in test_images if len(elem)<10] # filter file name
            num_test_images = len(test_images)

            for j in range(num_test_images):
                image_phone = misc.imread(input_path + test_images[j])
                fname = os.path.splitext(os.path.basename(test_images[j]))[0]
                
                image_phone = np.reshape(image_phone, [1, image_phone.shape[0], image_phone.shape[1], 3]) / 255
                enhanced = sess.run(output, feed_dict={x: image_phone})
                out_path = os.path.join(input_path, fname + "_enhanced_xU.png")
                misc.imsave(out_path, enhanced)
                #print(enhanced.shape)
                image_phone=np.reshape(image_phone,enhanced.shape)*255
                #print(image_phone.shape,enhanced.shape)
                before_after=np.hstack((image_phone,enhanced))
                #print(before_after.shape)
                #misc.imsave(os.path.join(input_path,fname+"_compare.png"),before_after)
