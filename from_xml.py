import cv2
import traceback

separators = ['<', '>', '/', ' ']

def get_label(row):
    lab = row.split('label=\"')[1].split('\" ')[0]
    return lab

def get_numbers(row):
    numlist = []
    tmp_num = ""
    found_number = False
    for c in row:
        if c >= '0' and c <= '9' or c == '.':
            found_number = True
        else:
            if found_number:
                numlist.append(float(tmp_num))
                tmp_num = ""
            found_number = False
        if found_number:
            tmp_num += c
    
    return numlist

labels = []

class Picture:

    def __init__(self, name, width, height):
        self.name = name
        self.annotations = []
        self.width = width
        self.height = height
    
    def add_annotation(self, data):
        self.annotations.append(data)

def find_image(text, file_type, path, valid_files, cam_prefix, export_dir):
    rows = text.split('\n')
    image_found = False
    tmp_img = None

    one_image_object = None

    test_img = None

    for row in rows:
        if '<image' in row:
            image_found = True
            img_string_start = row.split('name=\"')[1]
            resolution = img_string_start.split(file_type)
            picname = resolution[0].split('\" ')[0]
            picname = cam_prefix + picname + file_type
            #print(picname)
            
            try:
                
                test_img = cv2.imread(path + picname)
                
                h, w = test_img.shape[0], test_img.shape[1]

                one_image_object = Picture(picname, w, h)

                cv2.imwrite(export_dir + '/' + picname, test_img)
                
            except:

                image_found = False

            continue
            
        if image_found:
            if '<box' in row:
                
                x1, y1, x2, y2 = get_numbers(row.split('xtl')[1])
                label = get_label(row)
                #print(row)
                
                if label not in labels:
                    labels.append(label)
                
                
                one_image_object.add_annotation([x1, y1, x2, y2, label])
            
                
        
        if '</image' in row:
            if one_image_object != None:
                valid_files.append(one_image_object)
            one_image_object = None
            image_found = False

            

def read_one_xml(xml_filename, path, cam_prefix, export_dir):
    valid_files = []

    XML_FILE_NAME = xml_filename
    fopen = open(XML_FILE_NAME, 'r')
    fread = fopen.read()
    find_image(fread, '.jpg', path, valid_files, cam_prefix, export_dir)
    return valid_files


#read_one_xml('test.xml', './', 't873.cam1.', 'exported_images')



