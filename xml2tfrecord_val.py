import tensorflow as tf
import cv2
import numpy as np


# 獲取文件列表
files = tf.io.match_filenames_once("./image_label/xml/*.tfrecords")

# 創建輸入隊列
# 重複輸出輸入文件列表中的所有文件名，除非用參數num_epochs指定每個文件可輪詢的次數
# shuffle參數用於控制隨機打亂文件排序
# 返回值說明:
# A queue with the output strings. A QueueRunner for the Queue is added to the current Graph's QUEUE_RUNNER collection.
# 用tf.train.start_queue_runners啓動queue的輸出
queue = tf.train.string_input_producer(files, shuffle=True)


# 建立TFRecordReader並解析TFRecord文件
reader = tf.TFRecordReader()
_, serialized_example = reader.read(queue)  # tf.TFRecordReader.read()用於讀取queue中的下一個文件
rec_features = tf.parse_single_example(  # 返回字典，字典key值即features參數中的key值
    serialized_example,
    features={
        # 寫入時shape固定的數值用FixedLenFeature
        "filename": tf.FixedLenFeature(shape=[], dtype=tf.string),  # 由於只有1個值也可以用shape[1]，返回list
        "width": tf.FixedLenFeature(shape=[], dtype=tf.int64),
        "height": tf.FixedLenFeature(shape=[], dtype=tf.int64),
        "data": tf.FixedLenFeature(shape=[], dtype=tf.string),
        # 寫入時shape不固定的數值，讀出時用VarLenFeature，讀出爲SparseTensorValue類對象
        "object/label": tf.VarLenFeature(dtype=tf.string),
        "object/bbox/xmin": tf.VarLenFeature(dtype=tf.int64),
        "object/bbox/xmax": tf.VarLenFeature(dtype=tf.int64),
        "object/bbox/ymin": tf.VarLenFeature(dtype=tf.int64),
        "object/bbox/ymax": tf.VarLenFeature(dtype=tf.int64),
    }
)

# # 將tf.string轉化成tf.uint8的tensor
# img_tensor = tf.decode_raw(rec_features["data"], tf.uint8)
# print(img_tensor.shape)  # 輸出: dododo

with tf.Session() as sess:
    """
    sess.run(tf.global_variables_initializer())
    print(sess.run(files))
    上述代碼運行出錯，提示如下：
    Attempting to use uninitialized value matching_filenames
    因爲tf.train.match_filenames_once使用的是局部變量，非全局變量
    需要改成下方代碼才能正確運行
    """
    sess.run(tf.local_variables_initializer())
    print(sess.run(files))  # 打印文件列表

    # 用子線程啓動tf.train.string_input_producer生成的queue
    coord = tf.train.Coordinator()  # 用於控制線程結束
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 讀出TFRecord文件內容
    for i in range(20):
        # 每次run都由string_input_producer更新至下一個TFRecord文件
        rec = sess.run(rec_features)
        print(rec["filename"].decode("utf-8"))  # 由bytes類型轉爲str類型
        print("目標數目: " + str(rec["object/label"].values.size))
        print(rec["object/label"].values)
        print(rec["object/bbox/xmin"].values)
        print(rec["object/bbox/xmax"].values)
        print(rec["object/bbox/ymin"].values)
        print(rec["object/bbox/ymax"].values)
        
        # 將圖像數據轉化爲numpy.ndarray
        img = np.fromstring(rec["data"], np.uint8)
        print(type(rec["data"]))  # 輸出: <class 'bytes'>
        print(type(img))  # 輸出: <class 'numpy.ndarray'>

        # 根據feature設置圖像shape
        img = np.reshape(img, (rec["height"], rec["width"], 3))
        print(img.shape)  # 輸出: (rec["height"], rec["width"], 3)

        # 將圖像由RGB轉爲RGB用於imshow
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 繪製標註框
        for j in range(rec["object/label"].values.size):
            img = cv2.putText(img,
                              rec["object/label"].values[j].decode("utf-8"),
                              (rec["object/bbox/xmin"].values[j], rec["object/bbox/ymin"].values[j]-2),
                              cv2.FONT_HERSHEY_PLAIN,
                              1,
                              (0, 255, 0)
                              )
            img = cv2.rectangle(img,
                                (rec["object/bbox/xmin"].values[j], rec["object/bbox/ymin"].values[j]),
                                (rec["object/bbox/xmax"].values[j], rec["object/bbox/ymax"].values[j]),
                                (0, 0, 255))

        # 顯示圖像
        cv2.imshow(rec["filename"].decode("utf-8"), img)
        cv2.waitKey()

    coord.request_stop()  # 結束線程
    coord.join(threads)  # 等待線程結束


