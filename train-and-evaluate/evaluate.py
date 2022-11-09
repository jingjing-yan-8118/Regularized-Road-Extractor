from model import *
from road_network import *

import pickle
from subprocess import Popen
from time import time
import random
import json
from PIL import Image


save_folder='dataset/evaluate'
data_folder=''
run_time='run1'
learning_rate=0.0001
cnnBatchSize=128
noLeftRight=True
converage=False
step=0
step_max=300000
cnn_model='simple3'
use_batchnorm=True
lr_drop_interval=60000
output_prefix="dataset/evaluate"
config='dataset/sydney1/region_0_0/config.json'


validation_file='validation.p'
tiles_name='tiles'


Image.MAX_IMAGE_PIXELS = None
city='Wuhan'

modelname='test'

save_seg=False
# model_config='simpleCNN+GNNGRU_8_0_1_1'
model_config='simpleCNNonly'
model_recover='model/AerialKITTI/offset_mi_onlycnn/'+modelname
train_cnn_only=True
# train_cnn_only=False
# data_type='train'
data_type='test'
# gnn_model='RBDplusRawplusAux'
# gnn_model='ResGCNRBDplusRawplusAux'
gnn_model='GATRBDplusRawplusAux'


if __name__ == "__main__":
	dataset_folder = data_folder
	result_output = output_prefix
	image_list=list(os.listdir('../data_processing/%s/Segmentation/'%city))
	name_list=list(map(lambda x:x[:-4], image_list))
	config_root='../data_processing/%s/config/config_'%city
	data_root='../data_processing/%s/'%city

	test_configs=[]
	test_networks=[]

	# name_list=['150']
	if city=='Cheng':
		trainingSet=pickle.load(open("../data_processing/Cheng/training_data.p",'rb'))
		valiSet=pickle.load(open("../data_processing/Cheng/validation_data.p", "rb"))
		testSet=pickle.load( open("../data_processing/Cheng/testing_data.p", "rb"))
	if city=='Shaoxing':
		trainingSet=['sxa','sxb1','sxb2','sxc1']
		valiSet=['sxc2']
		testSet=['sxd']
	if city=='Wuhan':
		trainingSet=['a','c','d','e','g','h']
		valiSet=['f']
		testSet=['b']
	if city=='AerialKITTI':
		trainingSet=['0','1','2','3','4','5','6','7','8','9','10','11','2','13','14','15']
		valiSet=['16','17','18']
		testSet=['19','20']
	if city=='Bavaria':
		trainingSet=['L010','L080','L320','L105','M236','M040','M070','M090','M130']
		valiSet=['M176','M010']
		testSet=['M197']

	for name in name_list:
		config=json.load(open(config_root+name+'.json','r'))
		city_name = name

		if name in testSet or name in valiSet:
			test_configs.append(config_root+str(name)+'.json')

	for config_file in test_configs:
			print(config_file)
			config = json.load(open(config_file,"r"))
			name=config['name']
			roadNetwork = pickle.load(open(data_root+"roadnetwork/roadnetwork_"+name+'.p', "r"))
			# roadNetwork = pickle.load(open(data_root+"offsetdata_mi/roadnetwork_"+name+'.p', "r"))

			_,_ = roadNetwork.loadAnnotation(config_file, osm_auto=True,  root_folder = data_root)
			test_networks.append(roadNetwork)



	validation_set = []
	if validation_file is not None:

		try:
			validation_dict = pickle.load(open(validation_file,"r"))

			for k,v in validation_dict.iteritems():
				if k in config:
					validation_set = v 
					print("set validation set to ", k)
					print("size ", len(v))

		except:
			print("validation set not found")


	model_config = model_config

	cnn_type = "simple" # "simple"
	gnn_type = "simple" # "none"

	number_of_gnn_layer = 4
	remove_adjacent_matrix = 0

	GRU=False

	if model_config.startswith("simpleCNN+GNN"):
		gnn_type = gnn_model
		cnn_type = cnn_model
		sampleSize=32
		sampleNum=32
		stride=0

		items = model_config.split("_")

		number_of_gnn_layer = int(items[1])
		remove_adjacent_matrix = int(items[2]) # 0 or 1

		if model_config.startswith("simpleCNN+GNNGRU"):
			print("Use GRU")
			GRU = True 

	if model_config.startswith("simpleCNNonly"):

		gnn_type = "none"
		cnn_type = "simple"

		cnn_type = cnn_model

		print(model_config, cnn_type)

		sampleSize = 32
		sampleNum = 32
		stride = 0

		items = model_config.split("_")


	random.seed(123)

	config = tf.ConfigProto(device_count = {'GPU': 0})

	test_batch=1024

	for roadNetwork in test_networks:

		name=roadNetwork.name
		output_file=data_root+'output/annoresult/bothedge/cnngnn/%s/'%modelname
		Popen("mkdir -p %s" % (output_file), shell=True).wait()
		annoresult_root=data_root+'output/annoresult/bothedge/cnngnn/%s/'%modelname+name+'.p'
		anno={}
		nid2width={}
		nid2target={}
		single=0

		sst=0
		while sst<len(roadNetwork.nid2loc):
			print(sst)
			eed=min(len(roadNetwork.nid2loc),sst+test_batch)
			output_folder=data_root+'cnngnn/'
			input_region = SubRoadNetwork(roadNetwork,st=sst,ed=eed, output_folder = output_folder,tiles_name = tiles_name,  graph_size = 10000, search_mode = 0, augmentation=False, partial = True, remove_adjacent_matrix = remove_adjacent_matrix, reseed = True )
			# input_region = SubRoadNetwork(roadNetwork,st=sst,ed=eed, output_folder = data_root,tiles_name = tiles_name,  graph_size = 10000, search_mode = 0, augmentation=False, partial = True, remove_adjacent_matrix = remove_adjacent_matrix, reseed = True )

			random.seed(123)

			if train_cnn_only == True:
				node_feature = np.zeros((input_region.nonIntersectionNodeNum, 126))
				seg_output= np.zeros((input_region.nonIntersectionNodeNum, 64,64,1))
				outputs=np.zeros((input_region.nonIntersectionNodeNum, 4))
			else:
				node_feature = np.zeros((input_region.nonIntersectionNodeNum, 126))
				seg_output= np.zeros((input_region.nonIntersectionNodeNum, 64,64,1))
				outputs=np.zeros((input_region.nonIntersectionNodeNum, 4))



			print("Stage1")
			tf.reset_default_graph()
			time_cnn = 0
			# stage1 and stage2 can load 2 different model
			# stage 1
			with tf.Session() as sess:
				model = DeepRoadMetaInfoModel(sess, cnn_type, gnn_type, number_of_gnn_layer = number_of_gnn_layer, GRU=GRU, noLeftRight =noLeftRight, use_batchnorm = use_batchnorm)
				model.restoreModel(model_recover)
				if model.dumpWeights()==True:
					pass
				else:
					print("load model failed, nan encountered!!! try reloading the model!!!!! ")

				batch_size = cnnBatchSize # you may reduce this batch size if there is out-of-memory error.

				st = 0

				while st < input_region.nonIntersectionNodeNum:
					print(st)
					ed = min(st+batch_size, input_region.nonIntersectionNodeNum)

					t0 = time()
					node_feature[st:ed,:] = model.GetIntermediateNodeFeature(input_region, st, ed)[0]
					seg_output[st:ed,:] = model.GetSegOutput(input_region, st, ed)[0]
					time_cnn += time() - t0

					st += batch_size

				if save_seg==True:
					for i in range(input_region.nonIntersectionNodeNum):
						seg=(seg_output[i,:,:,:]*255).astype('int8')
						root=data_root+'cannyoutput/%s/tiles_'%data_type+roadNetwork.name
						Popen("mkdir -p %s" % root, shell=True).wait()

						cv2.imwrite(root+'/'+str(input_region.subGraphNoadList[i])+'.bmp',seg)

			print("Stage2")
			# stage 2
			tf.reset_default_graph()
			time_gnn = 0

			with tf.Session() as sess:


				model = DeepRoadMetaInfoModel(sess, cnn_type, gnn_type, number_of_gnn_layer = number_of_gnn_layer, GRU=GRU, noLeftRight =noLeftRight, use_batchnorm = use_batchnorm, stage = 2)
				model.restoreModel(model_recover)
				if model.dumpWeights()==True:
					pass
				else:
					print("load model failed, nan encountered!!! try reloading the model!!!!! ")
				st = 0
				_,outputs,_=model.EvaluateWithIntermediateNodeFeature(input_region, node_feature)

				rec_result = None #= osm_input_region.RecommendationAccNode(outputs, input_region, validation_set=validation_set)

			c=0
			#record test output(non intersection nodes)
			for node in input_region.subGraphNoadList[:input_region.nonIntersectionNodeNum]:
				w =int((outputs[c,0]-outputs[c,1]))
				w2=int((outputs[c,2]-outputs[c,3]))
				anno[node]={}
				anno[node]['d1']=w
				anno[node]['d2']=w2
				anno[node]['heading_vector_vertical']=input_region.annotation[node]['heading_vector_vertical']

				c+=1
			for node in input_region.subGraphNoadList[input_region.nonIntersectionNodeNum:]:
				anno[node]={}
				anno[node]['d1']=0
				anno[node]['d2']=0

			sst+=test_batch

		pickle.dump(anno, open(annoresult_root, "w"))


		


			


















			
