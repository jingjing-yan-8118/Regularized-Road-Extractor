import argparse
from model import *
from road_network import *

import pickle
from subprocess import Popen
import sys
from time import time, sleep
import random
import json
import os
import threading

from PIL import Image

import warnings

warnings.filterwarnings("ignore")

LeftRight = True
converage = False

save_folder = 'dataset/prefix'
data_folder = ''
step = 0
step_max = 300000
cnn_model = 'simple3'
# gnn_model='RBDplusRawplusAux'
# gnn_model='ResGCNRBDplusRawplusAux'
gnn_model = 'GATRBDplusRawplusAux'
hst = 0
crosscity = 0
dataset = 'config/dataset_sample.json'
Image.MAX_IMAGE_PIXELS = None

use_batchnorm = True
lr_drop_interval = 15000
use_node_drop = True
use_homogeneous_loss = True
tiles_name = 'tiles'

city = 'Wuhan'

data_root = '../data_processing/%s/' % city
# model_config='simpleCNN+GNNGRU_8_0_1_1'
model_config = 'simpleCNNonly'
# model_recover='model/AerialKITTI/offset_onlycnn/trainwidth'
model_recover = None
# train_cnn_only=False
train_cnn_only = True
learning_rate = 0.001
# train_gnn_only=True
train_gnn_only = False
homogeneous_loss_factor = 0.75

if __name__ == "__main__":
    random.seed(123)
    learning_rate = learning_rate
    model_config = model_config

    suffix = ""

    dropout_rate = 0.4
    loss_func = 'L2'
    loss = 'square'
    # loss_func='None'
    run_name = '1223_offset_onlycnn' + city
    target_size = '32'
    lr_inter = 8

    if train_gnn_only == True:
        suffix += "_trainGNNOnly_"

    if loss_func == 'L2':
        suffix += '_lossL2_'

    if model_recover != None:
        suffix += 'modelrecover'

    if train_cnn_only == True:
        model_folder = save_folder + model_config + suffix + "cnnonly_" + cnn_model  # filename
    else:
        model_folder = save_folder + model_config + suffix

    run = model_folder + "_" + run_name + "_lr_" + str(learning_rate) + '_inter_' + str(lr_inter)  # name
    model_folder = run

    log_folder = "alllogs/log"

    # Popen  --- make dir
    Popen("mkdir -p %s" % model_folder, shell=True).wait()
    Popen("mkdir -p %s" % (model_folder + "/validation"), shell=True).wait()
    Popen("mkdir -p %s" % log_folder, shell=True).wait()

    Popen("mkdir -p samples", shell=True).wait()
    Popen("mkdir -p test", shell=True).wait()
    Popen("mkdir -p validation", shell=True).wait()

    # default
    cnn_type = "resnet18"  # "simple"
    gnn_type = "simple"  # "none"

    number_of_gnn_layer = 4
    remove_adjacent_matrix = 0

    parking_weight = 0
    biking_weight = 0
    lane_weight = 1
    type_weight = 1

    GRU = False

    if model_config.startswith("simpleCNN+GNN"):
        gnn_type = gnn_model
        cnn_type = cnn_model
        # sampleSize = 64
        sampleSize = 32
        # sampleNum = 32
        sampleNum = 32
        # stride = 32
        stride = 0

        items = model_config.split("_")

        number_of_gnn_layer = int(items[1])
        remove_adjacent_matrix = int(items[2])  # 0 or 1

        if model_config.startswith("simpleCNN+GNNGRU"):
            print("Use GRU")
            GRU = True

    if model_config.startswith("simpleCNNonly"):

        gnn_type = "none"
        cnn_type = "simple"

        cnn_type = cnn_model

        print(model_config, cnn_type)

        # sampleSize = 128
        sampleSize = 32
        sampleNum = 32
        stride = 16

        items = model_config.split("_")

        if len(items) > 2:
            lane_weight = int(items[1])
            type_weight = int(items[2])

        if len(items) > 3:
            biking_weight = int(items[3])

    training_configs = []
    validation_configs = []

    image_list = list(os.listdir('../data_processing/%s/Segmentation/' % city))
    name_list = list(map(lambda x: x[:-4], image_list))
    config_root = '../data_processing/%s/config/config_' % city

    trainingSet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '2', '13', '14', '15']
    valiSet = ['16', '17', '18']
    testSet = ['19', '20']

    for name in name_list:
        config = json.load(open(config_root + name + '.json', 'r'))
        city_name = name
        if name in trainingSet:
            training_configs.append(config_root + str(name) + '.json')
        if name in valiSet:
            validation_configs.append(config_root + str(name) + '.json')

    factor = 0.99


    best_testing_acc = 0
    best_testing_acc_last_step = 0

    best_testing_loss = 1000.0
    best_testing_loss_last_step = 0

    best_training_loss = 1000.0
    best_training_loss_last_step = 0

    best_train_width_loss = 1000.0
    best_train_width_loss_last_step = 0

    best_test_width_loss = 1000.0
    best_test_width_loss_last_step = 0

    with tf.Session(config=tf.ConfigProto()) as sess:
        model = DeepRoadMetaInfoModel(sess, cnn_type, gnn_type,
                                      number_of_gnn_layer=number_of_gnn_layer, GRU=GRU, use_batchnorm=use_batchnorm,
                                      homogeneous_loss_factor=homogeneous_loss_factor, loss_func=loss_func,
                                      model_recover=model_recover)

        if model_recover is not None:
            model.restoreModel(model_recover)

        # loading dataset
        training_networks = []
        validation_networks = []
        # lane_numbers = [100]*6
        width_numbers = [100] * 32

        for config_file in training_configs:
            print(config_file)
            config = json.load(open(config_file, "r"))
            # output_folder = config["folder"]
            name = config['name']
            roadNetwork = pickle.load(open(data_root + "/offsetdata_mi/roadnetwork_" + name + '.p', "r"))
            width_numbers, _ = roadNetwork.loadAnnotation(config_file, osm_auto=True, width_numbers=width_numbers,
                                                          root_folder=data_root)
            training_networks.append(roadNetwork)

        for config_file in validation_configs:
            print(config_file)
            config = json.load(open(config_file, "r"))
            # output_folder = config["folder"]
            name = config['name']
            roadNetwork = pickle.load(open(data_root + "/offsetdata_mi/roadnetwork_" + name + '.p', "r"))
            width_numbers, _ = roadNetwork.loadAnnotation(config_file, osm_auto=True, width_numbers=width_numbers,
                                                          root_folder=data_root)
            validation_networks.append(roadNetwork)

        lane_balance_factor = [1.0 for x in width_numbers]

        writer = tf.summary.FileWriter(log_folder + "/" + run, sess.graph)

        Popen("pkill tensorboard", shell=True).wait()
        sleep(1)
        s_loss = 0
        s_train_homogeneous_loss = 0
        s_width_loss = 0
        s_detect_loss = 0
        s_cnn_width_loss = 0
        training_data = []
        train_op = model.train_op


        def loadTrainingDataAsyncBlock(st, ed):
            ining_data = []
            for i in range(st, ed):
                must_have_lane = True

                if i % 10 > 5:
                    must_have_lane = False

                batch_size = 64

                if cnn_model == "resnet18":
                    batch_size = 32

                must_have_lane_change = False

                if random.random() < 0.5:
                    must_have_lane_change = True

                tnid = random.randint(0, len(training_networks) - 1)

                if must_have_lane_change:
                    while len(training_networks[tnid].node_with_lane_change) == 0:
                        tnid = random.randint(0, len(training_networks) - 1)

                training_data.append(
                    SubRoadNetwork(training_networks[tnid], st=0, ed=len(training_networks[tnid].nid2loc),
                                   must_have_lane_change=must_have_lane_change, tiles_name=tiles_name,
                                   train_cnn_only=train_cnn_only, train_cnn_batch=batch_size, train_cnn_preload=256,
                                   graph_size=sampleSize + stride * random.randint(0, 2),
                                   search_mode=random.randint(0, 3), partial=True,
                                   remove_adjacent_matrix=remove_adjacent_matrix,
                                   output_folder=data_root + 'offsetdata_mi/',
                                   must_have_lane=must_have_lane))

            return training_data


        def loadTrainingDataAsync():
            p_num = 4
            training_data = []
            t0 = time()

            # td = [[] for x in range(p_num)]
            training_data = loadTrainingDataAsyncBlock(0, sampleNum)

            return training_data


        training_data = loadTrainingDataAsync()

        train_epoch_best_loss = 1000.
        no_optim = 0

        while True:
            if step == step_max + 1:
                break

            if step % 500 == 0 and step != 0:
                training_data = list()
                training_data = loadTrainingDataAsync()

            # if step % 1000 == 0 and step != 0:
            # 	model.saveModel(model_folder+"/model%d"%step)
            # 	model.saveCNNModel(model_folder+"/cnn_only_model%d"%step)

            loss_details = [0] * 6
            training_subgraph = training_data[random.randint(0, len(training_data) - 1)]

            if train_cnn_only == True:
                training_subgraph.RandomBatchST()

            batch_size = 64
            if train_cnn_only == False:
                batch_size = None

            loss, output_width, loss_width, _, train_homogeneous_loss, lossadd2, width_loss, detect_loss, cnn_width_loss = model.Train(
                training_subgraph, learning_rate=learning_rate, train_op=train_op, batch_size=batch_size,
                use_drop_node=use_node_drop, train_gnn_only=train_gnn_only, dropout=dropout_rate)

            s_loss += loss
            s_train_homogeneous_loss += train_homogeneous_loss
            s_cnn_width_loss += cnn_width_loss
            s_width_loss += width_loss
            s_detect_loss += detect_loss
            step += 1

            if step % 500 == 0:
                train_loss = s_loss / 500 + s_train_homogeneous_loss / 500
                if train_loss >= train_epoch_best_loss:
                    no_optim += 1
                    print('train_loss(%.3f)>best_loss(%.3f)' % (train_loss, train_epoch_best_loss))
                    print('no_optim=%f' % no_optim)
                else:
                    no_optim = 0
                    train_epoch_best_loss = train_loss
                    print('at step %d, best_loss=%.3f' % (step, train_epoch_best_loss))

                if no_optim > lr_inter:

                    if learning_rate < 5e-9:
                        pass
                    else:
                        learning_rate /= 3
                    no_optim = 0
                    print('at step %d change the lr to %.9f' % (step, learning_rate))
                test_loss = 0
                test_acc = 0
                test_homogeneous_loss = 0
                test_width_loss = 0
                test_detect_loss = 0
                test_cnn_width_loss = 0

                statistic_result = None
                ic = 0
                for rnk in validation_networks:
                    # test_sampleSize = 64
                    test_sampleSize = 128
                    test_sampleNum = 2
                    must_have_lane = True
                    subnet = SubRoadNetwork(rnk, st=0, ed=len(rnk.nid2loc), train_cnn_only=train_cnn_only,
                                            tiles_name=tiles_name,
                                            graph_size=test_sampleSize + 32 * random.randint(0, 2),
                                            search_mode=random.randint(0, 3), partial=True,
                                            remove_adjacent_matrix=remove_adjacent_matrix,
                                            output_folder=data_root + 'offsetdata_mi/',
                                            lane_balance_factor=lane_balance_factor, must_have_lane=must_have_lane,
                                            augmentation=False)
                    if train_cnn_only == True:
                        subnet.RandomBatchST()
                    outputs = model.Evaluate(subnet, batch_size=batch_size)
                    test_loss += outputs[0]
                    test_homogeneous_loss += outputs[3]
                    test_width_loss += outputs[5]
                    test_detect_loss += outputs[6]
                    test_cnn_width_loss += outputs[7]

                    if ic == len(
                            validation_networks) - 1 and step % 1000 == 0:  # every 1000 steps , record the last test graph
                        dump = True
                    else:
                        dump = False
                    test_acc_, statistic_result = subnet.GetAccuracyStatistic(outputs, statistic_result, dump=dump,
                                                                              batch_size=batch_size)
                    ic = ic + 1

                test_loss /= len(validation_networks)
                test_acc /= len(validation_networks)
                test_homogeneous_loss /= len(validation_networks)
                test_width_loss /= len(validation_networks)
                test_detect_loss /= len(validation_networks)
                test_cnn_width_loss /= len(validation_networks)

                if test_acc_ < best_testing_acc and step - best_testing_acc_last_step > 500:
                    print("New Best Model (testing acc) ", step, "acc", test_acc)

                    model.saveModelBest(model.saver_best1,
                                        model_folder + "/best_model%d_%d" % (step, int(test_acc * 1000)))
                    model.saveCNNModel(model_folder + "/cnn_only_best_model%d_%d" % (step, int(test_acc * 1000)))

                    best_testing_acc = test_acc_
                    best_testing_acc_last_step = step

                if train_loss < best_training_loss and step - best_training_loss_last_step > 500:
                    print("New Best Model (training)", step, train_loss)

                    model.saveModelBest(model.saver_best2,
                                        model_folder + "/best_model_training_%d_%d" % (step, int(train_loss * 1000)))

                    best_training_loss = train_loss
                    best_training_loss_last_step = step

                if test_loss < best_testing_loss and step - best_testing_loss_last_step > 500:
                    print("New Best Model (testing)", step, test_loss)

                    model.saveModelBest(model.saver_best3,
                                        model_folder + "/best_model_testing_%d_%d" % (step, int(test_loss * 1000)))

                    best_testing_loss = test_loss
                    best_testing_loss_last_step = step

                train_width_loss = s_width_loss / 500
                if train_width_loss < best_train_width_loss and step - best_train_width_loss_last_step > 500:
                    print("New Best Model (train_width_loss)", step, train_width_loss)

                    model.saveModelBest(model.saver_best4, model_folder + "/best_model_trainwidth_%d_%d" % (
                    step, int(train_width_loss * 1000)))

                    best_train_width_loss = train_width_loss
                    best_train_width_loss_last_step = step
                if test_width_loss < best_test_width_loss and step - best_test_width_loss_last_step > 500:
                    print("New Best Model (test_width_loss)", step, test_width_loss)

                    model.saveModelBest(model.saver_best5, model_folder + "/best_model_testwidth_%d_%d" % (
                    step, int(test_width_loss * 1000)))

                    best_test_width_loss = test_width_loss
                    best_test_width_loss_last_step = step

                print(
                "train/n step,test_acc,test_loss,train_loss,test_detect_loss,train_detect_loss,test_width_loss,train_width_loss,train_homoloss,train_loss_sum",
                step, test_acc_, test_loss, s_loss / 500, test_detect_loss, s_detect_loss / 500, test_width_loss,
                s_width_loss / 500, s_train_homogeneous_loss / 500, train_loss)

                summary = model.addLog(test_detect_loss=test_detect_loss,
                                       test_width_loss=test_width_loss,
                                       train_detect_loss=s_detect_loss / 500,
                                       train_width_loss=s_width_loss / 500,
                                       test_loss=test_loss,
                                       train_loss=s_loss / 500,
                                       total_train_loss=train_loss,
                                       train_homogeneous_loss=s_train_homogeneous_loss / 500,
                                       test_homogeneous_loss=test_homogeneous_loss,
                                       test_acc_overall=test_acc_,
                                       test_cnn_width_loss=test_cnn_width_loss,
                                       train_cnn_width_loss=s_cnn_width_loss / 500)
                writer.add_summary(summary, step)

                s_loss = 0
                s_train_homogeneous_loss = 0
                s_width_loss = 0
                s_detect_loss = 0
                s_cnn_width_loss = 0
