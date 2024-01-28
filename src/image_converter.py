import os
import numpy as np
from PIL import Image

DEFAULT_BIN_DIR = "./bin/ransome"
DEFAULT_OUTPUT_DIR = "./img"

def binary_to_grayscale_image(input_file, output_directory, target_image_size=(1024, 1024)):
    print(f"Try to convert {input_file} to {output_directory}")
    # 바이너리 파일 읽기
    with open(input_file, 'rb') as f:
        binary_data = f.read()

    # 바이너리 데이터의 길이 확인
    data_length = len(binary_data)

    # 적절한 이미지 크기 계산 (가로 길이는 일정하게 유지하고, 세로 길이는 데이터 길이에 따라 동적으로 조정)
    target_width = target_image_size[0]
    target_height = ((data_length + 1) // target_width)

    # 바이너리 데이터를 NumPy 배열로 변환 및 크기 조정
    binary_array = np.frombuffer(binary_data, dtype=np.uint8)
    image = binary_array[:target_width * target_height].reshape((target_height, target_width))

    # Grayscale 이미지로 변환
    grayscale_image = Image.fromarray(image, 'L')

    # 이미지 저장
    output_file_name = f"{os.path.splitext(os.path.basename(input_file))[0]}.png"
    output_file_path = os.path.join(output_directory, output_file_name)
    grayscale_image.save(output_file_path)

def convert(input_dir = DEFAULT_BIN_DIR, output_dir = DEFAULT_OUTPUT_DIR):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_list = os.listdir(input_dir)
    for file_name in file_list:
        input_file_path = os.path.join(input_dir, file_name)

        if os.path.splitext(file_name)[1] == ".exe":
            # Grayscale 이미지로 저장할 경로 설정
            output_image_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.png")

            # 바이너리를 Grayscale 이미지로 변환
            binary_to_grayscale_image(input_file_path, output_dir)

            print(f"Grayscale 이미지가 {output_image_path}에 저장되었습니다.")

def main():
    convert(DEFAULT_BIN_DIR, DEFAULT_OUTPUT_DIR)

if __name__ == '__main__':
    main()

    
