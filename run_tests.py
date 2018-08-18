# how to use
# python run_tests.py --n_tests 4 --n_images 2500
# this will print in a file called "tests.txt" the last line of all the tests

from subprocess import call, Popen, PIPE
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-nt", "--n_tests", type=int, required=True, default=4,
	help="number of tests to run")
ap.add_argument("-ni", "--n_images", type=int, required=False, default=2500,
	help="number of images per class to load")
args = vars(ap.parse_args())

splits = ["0.2", "0.3", "0.4", "0.5"] # si quiero mas pruebas, esta wea me va a impedir que se hagan las demas

train_command = ["python", "train_network.py", "--dataset", "../images/",
			"--model", "lenet_model.model", "--n_images", str(args["n_images"]), "--split", "split"]

last_line_command = train_command + ["|", "tail", "-1"]

# train_command[train_command.index("split")] # ubicacion de string "split"

output = []

for split in splits: # comentario linea 16
	cmd = train_command[train_command.index("split")].replace("split", split)
	output.append(Popen( cmd, stdout=PIPE ).communicate()[0].split('\n')[-1])
	
output = map(lambda x: x.replace('\r', ''), output)

f = open('tests.txt', 'w')

for i in range(len(output)):
	output[i] = splits[i] + " test_size, results: " + output[i]
	f.writeline(output[i])

f.close()

# for i in range(4):
# 	cmd = ["python", "asd.py", str(i), "|", "tail", "-1"]
# 	output.append(Popen( cmd, stdout=PIPE ).communicate()[0].split('\n')[-2])



