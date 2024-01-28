import image_converter as ic
import unzip as z

ZIP_DIR = "./rogues"
IMAGE_DIR = "./img/rogues"

def main():
    #z.unzip_files_in_directory(ZIP_DIR, z.ZIP_PASSWORD)
    ic.convert(ZIP_DIR, IMAGE_DIR)

if __name__ == "__main__":
    main()
