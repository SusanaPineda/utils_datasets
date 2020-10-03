import cv2, os, shutil

if __name__ == '__main__':
    # Parameters
    csv_file_path = "data/all.txt"  # Path(sys.argv[1])
    txt_files_path = "data/alltxt"

    with open(csv_file_path, "r") as file:
        # Discard the first and third lines and get headers from the second line
        file.readline()
        headers = file.readline()
        file.readline()

        # Create path for txts, overwrite if exists
        if (os.path.exists(txt_files_path)):
            shutil.rmtree(txt_files_path, ignore_errors=True)
        os.makedirs(txt_files_path)

        for row in file:
            rowsplit = row.split(',') # first name
            # Create txt
            f = open(txt_files_path + "/" + rowsplit[0].split('.')[0] + ".txt", "w")
            bounding_boxes = row.replace(']"', '').split(',"[')
            rowinit = "cone 0 0 0 "
            rowend = " 0 0 0 0 0 0 0\n"
            for bbox in bounding_boxes[1:]:
                x, y, h, w = list(map(int, bbox.replace(',', '').replace('\n', '').split(' ')))
                f.write(row + str(x) + ' ' + str(y) + ' ' + str(x+w) + ' ' + str(y+h) + rowend)
