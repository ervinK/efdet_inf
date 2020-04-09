
import argparse
from datetime import date
import os
import sys
import tensorflow as tf
import from_xml
import inference

# import keras
# import keras.preprocessing.image
# import keras.backend as K
# from keras.optimizers import Adam, SGD

from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, SGD

from augmentor.color import VisualEffect
from augmentor.misc import MiscEffect
from model import efficientdet
from losses import smooth_l1, focal, smooth_l1_quad
from efficientnet import BASE_WEIGHTS_PATH, WEIGHTS_HASHES


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def get_session():
    """
    Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_callbacks(path):
    """
    Creates the callbacks to use during training.

    Args
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []
    

    # save the model
    
    # ensure directory created first; otherwise h5py will error after epoch.
    
    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(
            path,
            'efdet_model.h5' 
        ),
        verbose=1,
        save_weights_only=True,
        # save_best_only=True,
        # monitor="mAP",
        # mode='max'
    )
    callbacks.append(checkpoint)

    # callbacks.append(keras.callbacks.ReduceLROnPlateau(
    #     monitor='loss',
    #     factor=0.1,
    #     patience=2,
    #     verbose=1,
    #     mode='auto',
    #     min_delta=0.0001,
    #     cooldown=0,
    #     min_lr=0
    # ))

    return callbacks


def create_generators(batch_size, phi, is_text_detect, is_detect_quadrangle, rand_transf_augm, train_ann_path, val_ann_path, train_class_path, val_class_path):
    """
    Create generators for training and validation.

    Args
        args: parseargs object containing configuration for generators.
        preprocess_image: Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size': batch_size,
        'phi': phi,
        'detect_text': is_text_detect,
        'detect_quadrangle': is_detect_quadrangle
    }

    # create random transform generator for augmenting training data
    if rand_transf_augm:
        misc_effect = MiscEffect()
        visual_effect = VisualEffect()
    else:
        misc_effect = None
        visual_effect = None

    
    from generators.csv_ import CSVGenerator
    train_generator = CSVGenerator(
        train_ann_path,
        train_class_path,
        misc_effect=misc_effect,
        visual_effect=visual_effect,
        **common_args
    )

    if val_ann_path:
        validation_generator = CSVGenerator(
            val_ann_path,
            val_class_path,
            shuffle_groups=False,
            **common_args
        )
    else:
        validation_generator = None

    return train_generator, validation_generator

TRAIN_DATA_PATH = '../../data/train/'
VALID_DATA_PATH = '../../data/val/'
TEST_DATA_PATH = '../../data/test/'

idx = 0 #globalis valtozo, ezzel adunk a parsolt xml-ben id-t a kepeknek.

def get_class_types_from_data(TRAIN_DATA_PATH):
    """
    Megkapjuk a tanitohalmaz eleresi utvonalat, innen kinyerjuk
    hogy hany es milyen osztalyaink vannak

    return: osztalyok neveit tartalmazo n elemu lista
    """
    classes = []
    files = os.listdir(TRAIN_DATA_PATH)
    for f in files:
        if os.path.isdir(TRAIN_DATA_PATH + f):
            classes.append(f)
    return classes

def parse_to_one_xml(TRAIN_DATA_PATH, classes, pic_collector_folder):
    if not os.path.isdir(pic_collector_folder):
        os.mkdir(pic_collector_folder)
    """
    Az altlam megirt parser segitsegevel kinyerjuk az xml fajlokbol az annotaciokat 
    es a parsolashoz szukseges tobbi informaciot

    return Kep objecteket ad vissza egy listaban, amik minden szuksegges informaciot tartalmaznak
    """
    collector = []
    for c in classes:
        xml_file_path = TRAIN_DATA_PATH + str(c) + '/' + c + '.xml'
        result = from_xml.read_one_xml(xml_file_path, TRAIN_DATA_PATH + str(c) + '/', '', pic_collector_folder)
        for x in result:
            collector.append(x)
    return collector

def preproc_xml(fwriter, class_types):
    """
    Kiiratjuk az xml metaadatait
    """

    fwriter.write('<?xml version="1.0" encoding="UTF-8"?>' + '\n')
    fwriter.write('<annotations>' + '\n')
    fwriter.write('\t' + '<version>1.1</version>' + '\n')
    fwriter.write('\t' + '<meta>' + '\n')
    fwriter.write('\t\t' + '<mode>annotation</mode>' + '\n')
    fwriter.write('\t\t' + '<labels>' + '\n')
    for c in class_types:
        fwriter.write('\t\t\t' + '<label>' + '\n')
        fwriter.write('\t\t\t\t' + '<name>' + str(c) + '</name>' + '\n')
        fwriter.write('\t\t\t' + '</label>' + '\n')
    fwriter.write('\t\t' + '</labels>' + '\n')
    fwriter.write('\t' + '</meta>' + '\n')

def create_limits_txt(class_types):
    """
    Elkeszitjuk a limits txt-t a kulonbozo osztalyoknak
    """
    limit_writer = open('classes.csv','w')
    for c in range(0, len(class_types)):
        limit_writer.write(str(class_types[c]) + ',' + str(c) + '\n')

def write_pic(result, fwriter):
    global idx
    for pic in result:
        fwriter.write('\t' + '<image height=\"' + str(pic.height) + '\" width=\"' + str(pic.width) + '\" name=\"' + str(pic.name) + '\" id=\"' + str(idx) + '\">' + '\n')
        for box in pic.annotations:
            fwriter.write('\t\t' + '<box ybr=\"' + str(box[3]) + '\" xbr=\"' + str(box[2]) + '\" ytl=\"' + str(box[1]) + '\" xtl=\"' + str(box[0]) + '\" occluded=\"0\" label=\"' + str(box[4]) + '\"></box>' + '\n')
        fwriter.write('\t' + '</image>' + '\n')
        print(idx)
        idx += 1
    fwriter.write('</annotations>')

def write_ann(result, fwriter, mode):
    for pic in result:
        for box in pic.annotations:
            fwriter.write(str(mode) + '/' + pic.name + ',' + str(int(box[0])) + ',' + str(int(box[1])) + ',' + str(int(box[2])) + ',' + str(int(box[3])) + ',' + str(box[4]) + '\n')



def main(train_csv, val_csv, classes_csv, epoch_num, phi_num, steps_epoch, batch_num, model_path):


    batch_size = batch_num #def 1
    phi = phi_num #def 0
    is_text_detect = False
    is_detect_quadrangle = False
    rand_transf_augm = True
    train_ann_path = train_csv
    train_class_path = classes_csv
    val_ann_path = val_csv
    val_class_path = classes_csv
    epochs = epoch_num
    workers = 1
    steps_p_epoch = steps_epoch
    use_multiproc = True
    max_que_size = 10
    comp_loss = True
    gpu = 0
    freeze_bn_arg = True
    weighted_bi = False

    # create the generators
    train_generator, validation_generator = create_generators(batch_size, phi, is_text_detect, is_detect_quadrangle, rand_transf_augm, train_ann_path, val_ann_path, train_class_path, val_class_path)

    num_classes = train_generator.num_classes()
    num_anchors = train_generator.num_anchors

    # optionally choose specific GPU
    if gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    # K.set_session(get_session())

    model, prediction_model = efficientdet(phi,
                                           num_classes=num_classes,
                                           num_anchors=num_anchors,
                                           weighted_bifpn=weighted_bi,
                                           freeze_bn=freeze_bn_arg,
                                           detect_quadrangle=is_detect_quadrangle
                                           )
    
    # freeze backbone layers
    if freeze_bn_arg:
        # 227, 329, 329, 374, 464, 566, 656
        for i in range(1, [227, 329, 329, 374, 464, 566, 656][phi]):
            model.layers[i].trainable = False


    model.compile(optimizer=Adam(lr=1e-3), loss={
        'regression': smooth_l1_quad() if is_detect_quadrangle else smooth_l1(),
        'classification': focal()
    }, )
    
    # print(model.summary())

    # create the callbacks
    callbacks = create_callbacks(model_path)


    if not comp_loss:
        validation_generator = None
    elif comp_loss and validation_generator is None:
        raise ValueError('When you have no validation data, you should not specify --compute-val-loss.')
    
    # start training
    return model.fit_generator(
        generator=train_generator,
        steps_per_epoch=steps_p_epoch,
        initial_epoch=0,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        workers=workers,
        use_multiprocessing=use_multiproc,
        max_queue_size=max_que_size,
        validation_data=validation_generator
    )


if __name__ == '__main__':
    class_types = get_class_types_from_data(TRAIN_DATA_PATH)

    create_limits_txt(class_types)

    

    train_val_dataset = []

    train_val_dataset += parse_to_one_xml(TRAIN_DATA_PATH, class_types, 'all_images') 


    fwrite = open('cvat_parsed.xml', 'w')

    preproc_xml(fwrite, class_types)
    write_pic(train_val_dataset, fwrite)

    fwrite = open('cvat_parsed.txt', 'w')

    preproc_xml(fwrite, class_types)
    write_pic(train_val_dataset, fwrite)

    fwrite = open('train.csv', 'w')
    write_ann(train_val_dataset, fwrite, 'all_images')

    fwrite = open('val.csv', 'w')
    write_ann(train_val_dataset, fwrite, 'all_images')

    batch_num = 1
    steps_per_epoch = 1
    phi_num = 0
    epochs = 5

    model_path = '../../models'

    main('train.csv', 'val.csv', 'classes.csv', epochs, phi_num, steps_per_epoch, batch_num, model_path)

    #inference.inf(phi_num, False, len(class_types))