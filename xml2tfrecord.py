import tensorflow as tf  # 導入TensorFlow
import cv2  # 導入OpenCV
import os  # 用於文件操作
import glob  # 用於遍歷文件夾內的xml文件
import xml.etree.ElementTree as ET  # 用於解析xml文件


# 將LabelImg標註的圖像文件和標註信息保存爲TFRecord
class LabelImg2TFRecord:

    @classmethod
    def gen(cls, path):
        """
        :param path: LabelImg標識文件的路徑，及生成的TFRecord文件路徑
        """
        # 遍歷文件夾內的全部xml文件，1個xml文件描述1個圖像文件的標註信息
        for f in glob.glob(path + "/*.xml"):
            # 解析xml文件
            try:
                tree = ET.parse(f)
            except FileNotFoundError:
                print("無法找到xml文件: "+f)
                return False
            except ET.ParseError:
                print("無法解析xml文件: "+f)
                return False
            else:  # ET.parse()正確運行

                # 取得xml根節點
                root = tree.getroot()

                # 取得圖像路徑和文件名
                img_name = root.find("filename").text
                img_path = root.find("path").text

                # 取得圖像寬高
                img_width = int(root.find("size")[0].text)
                img_height = int(root.find("size")[1].text)

                # 取得所有標註object的信息
                label = []  # 類別名稱
                xmin = []
                xmax = []
                ymin = []
                ymax = []

                # 查找根節點下全部名爲object的節點
                for m in root.findall("object"):
                    xmin.append(int(m[4][0].text))
                    xmax.append(int(m[4][2].text))
                    ymin.append(int(m[4][1].text))
                    ymax.append(int(m[4][3].text))
                    # 用encode將str類型轉爲bytes類型，相應的用decode由bytes轉回str類型
                    label.append(m[0].text.encode("utf-8"))

                # 至少有1個標註目標
                if len(label) > 0:
                    # 用OpenCV讀出圖像原始數據，未壓縮數據
                    data = cv2.imread(img_path, cv2.IMREAD_COLOR)

                    # 將OpenCV的BGR格式轉爲RGB格式
                    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

                    # 建立Example
                    example = tf.train.Example(features=tf.train.Features(feature={
                        # 用encode將str類型轉爲bytes類型
                        # 以下各feature的shape固定，讀出時必須使用tf.FixedLenFeature
                        "filename": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_name.encode("utf-8")])),
                        "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[img_width])),
                        "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[img_height])),
                        "data": tf.train.Feature(bytes_list=tf.train.BytesList(value=[data.tostring()])),  # 圖像數據ndarray轉化成bytes類型
                        # 以下各feature的shape不固定，，讀出時必須使用tf.VarLenFeature
                        "object/label": tf.train.Feature(bytes_list=tf.train.BytesList(value=label)),
                        "object/bbox/xmin": tf.train.Feature(int64_list=tf.train.Int64List(value=xmin)),
                        "object/bbox/xmax": tf.train.Feature(int64_list=tf.train.Int64List(value=xmax)),
                        "object/bbox/ymin": tf.train.Feature(int64_list=tf.train.Int64List(value=ymin)),
                        "object/bbox/ymax": tf.train.Feature(int64_list=tf.train.Int64List(value=ymax))
                    }))

                    # 建立TFRecord的寫對象
                    # img_name.split('.')[0]用於去掉擴展名，只保留文件名
                    with tf.io.TFRecordWriter(os.path.join(path, img_name.split('.')[0]+".tfrecords")) as writer:
                        # 數據寫入TFRecord文件
                        writer.write(example.SerializeToString())

                        # 結束
                        print("生成TFRecord文件: " + os.path.join(path, img_name.split('.')[0]+".tfrecords"))
                else:
                    print("xml文件{0}無標註目標".format(f))
                    return False

        print("完成全部xml標註文件的保存")
        return True


if __name__ == "__main__":
    LabelImg2TFRecord.gen("./image_label/xml")

