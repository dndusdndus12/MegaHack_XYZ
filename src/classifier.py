import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

TEST_DIR = "./new_img"

GPU_MODE = True

DEFAULT_IMAGE_SIZE = (64, 64)
DEFAULT_COPR_SIZE = (32, 32) #이미지를 겹치게 생성

def crop_and_preprocess_data(data_directory, categories, crop_size=DEFAULT_COPR_SIZE, target_size=DEFAULT_IMAGE_SIZE):
    data = []
    labels = []
    num_of_img = 0

    print(f"Image Fragmentation Size: ({DEFAULT_IMAGE_SIZE[0]}, {DEFAULT_IMAGE_SIZE[1]})")

    for category_id, category in enumerate(categories):
        category_path = os.path.join(data_directory, category)
        file_names = os.listdir(category_path)
        num_of_img += len(file_names)

        print(f"Load Images({len(file_names)}) from {category_path}")
        for file_name in file_names:
            # print(f"Load file: {file_name}")
            file_path = os.path.join(category_path, file_name)
            original_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            # 이미지를 crop_size로 여러 부분으로 나누어 사용
            for i in range(0, original_image.shape[0] - crop_size[0], crop_size[0] // 2):
                for j in range(0, original_image.shape[1] - crop_size[1], crop_size[1] // 2):
                    cropped_image = original_image[i:i+crop_size[0], j:j+crop_size[1]]
                    
                    # 이미지 크기를 모델 입력에 맞게 조정
                    resized_image = cv2.resize(cropped_image, target_size)

                    data.append(resized_image)
                    labels.append(category_id)

    data = np.array(data).reshape(-1, target_size[0], target_size[1], 1)
    labels = np.array(labels)
    print(f"Number of malware: {num_of_img}\nNumber of input: {len(data)}, {len(labels)}")
    return data, labels

def main():
    # 데이터 경로 설정
    data_directory = TEST_DIR  # 이미지 데이터가 들어있는 상위 폴더
    categories = os.listdir(data_directory)
    num_classes = len(categories)

    data, labels = crop_and_preprocess_data(data_directory, categories)
    data = np.array(data).reshape(-1, DEFAULT_IMAGE_SIZE[0], DEFAULT_IMAGE_SIZE[1], 1)
    labels = np.array(labels)

    # 데이터 분할
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

    # 모델 정의
    model = models.Sequential([
        # layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(DEFAULT_IMAGE_SIZE[0], DEFAULT_IMAGE_SIZE[1], 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 데이터 증강을 통한 학습
    datagen = ImageDataGenerator(
        width_shift_range=0.2,  # 수평으로 최대 20% 이동
        height_shift_range=0.2,  # 수직으로 최대 20% 이동
        shear_range=0.2,  # shear 적용
        fill_mode='nearest'  # 새로운 픽셀을 채울 때 가장 가까운 픽셀을 사용
    )

    datagen.fit(train_data)


    # 모델 학습
    model.fit(datagen.flow(train_data, train_labels, batch_size=64), epochs=10, validation_data=(test_data, test_labels))

    # 모델 평가
    test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
    print(f"\nTest accuracy: {test_acc}")


def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


if __name__ == '__main__':
    print(get_available_devices())
    if GPU_MODE:
        # GPU setup
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    main()

