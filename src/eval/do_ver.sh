
#python -u verification.py --gpu 0 --data-dir /opt/jiaguo/faces_vgg_112x112 --image-size 112,112 --model '../../model/softmax1010d3-r101-p0_0_96_112_0,21|22|32' --target agedb_30
python -u verification.py --gpu 0 --data-dir /home/ubuntu/camel/workspace/data/faces_emore --model '/home/ubuntu/camel/workspace/projects/insightface/recognition/models/20190415171744/r50-svxface-emore_glint/models,0' --target lfw,cfp_ff,cfp_fp,agedb_30,surveillance --batch-size 64
