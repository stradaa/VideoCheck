import subprocess

rec = "010"
path_avi = "/vol/brains/bd1/pesaranlab/Troopa_Thalamus_Ephys_Behave1/251030/"+rec+"/rec"+rec+".bag/img/Cam4/"
path_out = "/vol/brains/bd2/pesaranlab/Lab_People/Alex/pyfun/VideoCheck/vids_251030/"
# print(path_out)

subprocess.run([
    "python3", "trim_avi_clips.py",
    path_avi,
    path_out,
    "--duration", "20",
    "--overwrite",
], check=True)
