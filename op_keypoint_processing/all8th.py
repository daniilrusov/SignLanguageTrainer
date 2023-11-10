import os
import cv2

def extract_and_resize_frames(input_folder, output_folder, frame_interval=8, target_side_length=1366):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Extract frames
            cap = cv2.VideoCapture(input_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Determine the video's orientation
            if frame_width > frame_height:
                # Landscape orientation
                target_width = target_side_length
                target_height = int(target_side_length * frame_height / frame_width)
            else:
                # Portrait orientation
                target_height = target_side_length
                target_width = int(target_side_length * frame_width / frame_height)

            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (target_width, target_height))

            for i in range(frame_count):
                ret, frame = cap.read()
                if i % frame_interval == 0:
                    # Resize the frame based on orientation
                    frame = cv2.resize(frame, (target_width, target_height))
                    out.write(frame)

            cap.release()
            out.release()

if __name__ == "__main__":
    input_folder = r'C:\Users\1\Desktop\archive\slovo\all_time_test'
    output_folder = r'C:\Users\1\Desktop\archive\slovo\all8th_time_test'
    frame_interval = 8
    target_side_length = 1280

    extract_and_resize_frames(input_folder, output_folder, frame_interval, target_side_length)

if __name__ == "__main__":
    input_folder = r'C:\Users\1\Desktop\archive\slovo\all_fam_test'
    output_folder = r'C:\Users\1\Desktop\archive\slovo\all8th_fam_test'
    frame_interval = 8
    target_side_length = 1280

    extract_and_resize_frames(input_folder, output_folder, frame_interval, target_side_length)


