# import subprocess
#
# # wave_location = input()
# wave_location = r"D:\whisper\1.wav"
#
# result = subprocess.run(f"whisper {wave_location}", shell=True)
# print(result.stdout)


import subprocess

# waveName = r"D:\whisper\1.wav"
# outputFileName = r"D:\whisper\1.txt"
# command = f"""whisper {waveName} > {outputFileName}\n"""
#
# # whisper D:\whisper\1.wav > D:\whisper\1.txt
#
#
# with open('run.bat', 'w') as fileOut:
#     fileOut.write(command)

subprocess.run("run.bat")
