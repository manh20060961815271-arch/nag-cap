import os
import numpy as np
import tensorflow.compat.v1 as tf

import Add_Layer
import Data_Processing

tf.disable_v2_behavior()

def testing(data_test, window_size, first_hid, second_hid, scaler_X, scaler_Y):
    tf.reset_default_graph()
    BATCH_SIZE = None
    
    xs = tf.placeholder(tf.float32, [BATCH_SIZE, window_size, 5])

    # 1. Khai báo kiến trúc mạng 3 lớp khớp hoàn toàn với tệp Training.py
    (W_1, V_1, l1) = Add_Layer.add_layer(xs, [2, window_size, 5], first_hid, 
                                         activation_function=tf.nn.relu, use_layernorm=True, layer_name="layer1")
    (W_2, V_2, l2) = Add_Layer.add_layer(l1, first_hid, second_hid, 
                                         activation_function=tf.nn.relu, use_layernorm=True, layer_name="layer2")
    (W_3, V_3, prediction) = Add_Layer.add_layer(l2, second_hid, [BATCH_SIZE, 1, 2], 
                                                 activation_function=None, use_layernorm=False, layer_name="layer3")

    sess = tf.Session()
    saver = tf.compat.v1.train.Saver(tf.trainable_variables())
    
    save_dir = r'C:\tmp_cyclone_model'
    save_path = os.path.join(save_dir, 'model')
    
    # Khôi phục mô hình đã huấn luyện thành công từ checkpoint
    saver.restore(sess, save_path)

    Y_pred, Y_true = [], []
    a_list, b_list = [], []

    for storm in data_test:
        X, Y = Data_Processing.data_processing(storm, window_size)
        if len(X) == 0:
            continue
            
        Y_true_tmp = np.array(Y, dtype=np.float64).reshape(-1, 2)
        Y_true.append(Y_true_tmp)
        
        X_scaled = scaler_X.transform(np.array(X, dtype=np.float64).reshape(-1, 5)).reshape(-1, window_size, 5)
        
        # Chạy suy luận trực tiếp qua mạng không cần vòng lặp bộ nhớ
        pred_scaled = sess.run(prediction, feed_dict={xs: X_scaled})
        
        storm_preds_np = np.array(pred_scaled).reshape(-1, 2)
        Y_pred_tmp = scaler_Y.inverse_transform(storm_preds_np)
        
        Y_pred.append(Y_pred_tmp)
        a_list.append(Y_pred_tmp)
        b_list.append(Y_true_tmp)

    a = np.concatenate(a_list, axis=0) if a_list else np.array([])
    b = np.concatenate(b_list, axis=0) if b_list else np.array([])
    
    sess.close()
    return a, b, Y_true, Y_pred