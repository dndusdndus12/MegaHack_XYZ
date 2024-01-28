import os
import zipfile

DIR = "./ransomwares"
ZIP_PASSWORD = "mysubsarethebest"

def unzip_files_in_directory(directory, password):
    # 디렉토리 내의 모든 파일 목록 가져오기
    file_list = os.listdir(directory)

    # zip 파일만 선택
    zip_files = [file for file in file_list if file.endswith(".zip")]

    # 각 zip 파일에 대해 압축 해제 시도
    for zip_file in zip_files:
        zip_file_path = os.path.join(directory, zip_file)

        try:
            # Zip 파일 열기
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                # 비밀번호를 사용하여 압축 해제
                zip_ref.extractall(directory, pwd=password.encode('utf-8'))
                print(f"{zip_file}을 압축 해제하였습니다.")

        except zipfile.BadZipFile:
            print(f"주의: {zip_file}은 유효한 Zip 파일이 아닙니다.")

        except zipfile.LargeZipFile:
            print(f"주의: {zip_file}은 크기가 너무 큰 Zip 파일입니다.")
        
        except Exception as e:
            print(f"Unexpected Exception Occurred: {e}")
            print(f"at {zip_file}")

def main():
    # 압축 해제할 디렉토리 및 비밀번호 설정
    target_directory = DIR
    zip_password = ZIP_PASSWORD

    # 디렉토리 내의 zip 파일 압축 해제
    unzip_files_in_directory(target_directory, zip_password)

if __name__ == "__main__":
    main()
