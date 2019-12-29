import sys
import os
from scipy import misc
import imageio
from tqdm import tqdm

input_dir = sys.argv[1]
output_dir = sys.argv[2]

output_filename = os.path.join(output_dir, 'train.lst')
try:
    outFile = open(output_filename, "w")
except IOError:
    print("output file does not exist")

label = 0
labelToNameFile = open("label_to_name.txt", "w")
for person_name in tqdm(os.listdir(input_dir)):
    personDir = os.path.join(input_dir, person_name)
    incrLabel = False
    for person in tqdm(os.listdir(personDir)):
        imgPath = os.path.join(personDir, person)
        if not os.path.exists(imgPath):
            print('image not found (%s)'%imgPath)
            continue
        try:
            img = imageio.imread(imgPath)
        except (IOError, ValueError, IndexError) as e:
            errorMessage = '{}: {}'.format(imgPath, e)
            print(errorMessage)
        else:
            if img.ndim<2:
                print('Unable to align "%s", img dim error' % imgPath)
                #text_file.write('%s\n' % (output_filename))
                continue
            oline = '%d\t%s\t%d\n' % (1, imgPath, label)
            outFile.write(oline)        
            incrLabel = True
    if incrLabel:
        labelToNameFile.write("%d, %s\n" % (label, person_name))
        label += 1

outFile.close()
labelToNameFile.close()