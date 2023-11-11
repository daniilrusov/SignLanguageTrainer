import pandas as pd
import os
import subprocess
filtered_df_time=pd.read_excel(r'C:\Users\1\Desktop\archive\slovo\filtered_df_time.xlsx')
filtered_df_family=pd.read_excel(r'C:\Users\1\Desktop\archive\slovo\filtered_df_family.xlsx')

video_list = list(filtered_df_time.iloc[13:].attachment_id)
output_folder = f"C:/Users/1/Desktop/archive/slovo/all8th_time/"
print(video_list)


for video_name in video_list:
    video_path = os.path.join(output_folder, f"{video_name}.mp4")
    json_output_folder = os.path.join(output_folder, video_name)

    command = f"C:/Users/1/Desktop/openpose/bin/OpenPoseDemo.exe --video {video_path} --hand --display 0 --render_pose 0 --model_folder C:/Users/1/Desktop/openpose/models --write_json {json_output_folder}"

    subprocess.call(command, shell=True)

print("done")