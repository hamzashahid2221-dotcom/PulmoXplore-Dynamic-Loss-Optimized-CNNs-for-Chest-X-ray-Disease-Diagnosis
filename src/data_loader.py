import pathlib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet import preprocess_input


def load_data_paths():
    base_path1='/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL'
    base_path2='/kaggle/input/tuberculosis-tb-chest-xray-dataset/TB_Chest_Radiography_Database/Normal'
    base_path3='/kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset/Normal/images'
    normal_base_path=[base_path1,base_path2,base_path3]

    base_path4='/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA'
    base_path5='/kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset/Viral Pneumonia/images'
    pneumonia_base_path=[base_path4,base_path5]

    tub_base_path='/kaggle/input/tuberculosis-tb-chest-xray-dataset/TB_Chest_Radiography_Database/Tuberculosis'
    tuberculosis_base_path=[tub_base_path]

    selected_path=[normal_base_path,pneumonia_base_path,tuberculosis_base_path]
    selected_classes=['Normal','PNEUMONIA','Tuberculosis']

    valid_exts = [".png", ".jpg", ".jpeg"]
    all_images_path=[]
    all_labels=[]

    for idx,cls_path in enumerate(selected_path):
        for path in cls_path:
            cls_dir=pathlib.Path(path)
            images = [img for img in cls_dir.glob("*") if img.suffix.lower() in valid_exts]
            all_images_path.extend([str(img) for img in images])
            all_labels.extend([idx]*len(images))

    return np.array(all_images_path), np.array(all_labels), selected_classes


def preprocess(image,label,num_classes):
    img=tf.io.read_file(image)
    img=tf.io.decode_image(img,channels=3,expand_animations=False)
    img=tf.image.resize(img,[224,224])
    img=preprocess_input(img)
    return img,tf.one_hot(label,depth=num_classes)


def get_datasets(batch_size=64):
    all_images_path, all_labels, selected_classes = load_data_paths()

    train_path,temp_path,train_labels,temp_labels=train_test_split(
        all_images_path,all_labels,
        test_size=0.3,
        random_state=42,
        stratify=all_labels)

    val_path,test_path,val_labels,test_labels=train_test_split(
        temp_path,temp_labels,
        test_size=0.5,
        stratify=temp_labels)

    train_ds=tf.data.Dataset.from_tensor_slices((train_path,train_labels))
    train_ds=train_ds.map(lambda x,y: preprocess(x,y,len(selected_classes)),
                          num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds=tf.data.Dataset.from_tensor_slices((val_path,val_labels))
    val_ds=val_ds.map(lambda x,y: preprocess(x,y,len(selected_classes)),
                      num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_ds=tf.data.Dataset.from_tensor_slices((test_path,test_labels))
    test_ds=test_ds.map(lambda x,y: preprocess(x,y,len(selected_classes)),
                        num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, selected_classes
