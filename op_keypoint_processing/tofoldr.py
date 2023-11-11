import pandas as pd
import os
df = pd.read_csv(r'C:\Users\1\Desktop\archive\slovo\annotations.csv', sep='\t')

top_texts = df['text'].value_counts().head(10).index

TIME_PACKAGE  = [
    "no_event",
    "минута",
    "часы",
    "год",
    "день",
    "месяц",
    "вечер",
    "утро",
    "завтра",
    "сегодня",
]
FAMILY_PACKAGE = [
    "no_event",
    "отец",
    "семья",
    "сын",
    "дочь",
    "жена",
    "брат",
    "сестра",
]

filtered_df_test = df[(df['text'].isin(top_texts)) & (df['train'] == False)].sort_values(by='text').reset_index(drop=True)
filtered_df_test = filtered_df_test.iloc[285:]

filtered_df_time_test = df[(df['text'].isin(TIME_PACKAGE)) & (df['train'] == False)].sort_values(by='text').reset_index(drop=True)
filtered_df_time_test = filtered_df_time_test.iloc[95:]
filtered_df_family_test = df[(df['text'].isin(FAMILY_PACKAGE)) & (df['train'] == False)].sort_values(by='text').reset_index(drop=True)
filtered_df_family_test = filtered_df_family_test.iloc[95:]
print(filtered_df_family_test.text.value_counts())
print(filtered_df_time_test.text.value_counts())

filtered_df_test.to_excel(r'C:\Users\1\Desktop\archive\slovo\filtered_df_test.xlsx')
filtered_df_time_test.to_excel(r'C:\Users\1\Desktop\archive\slovo\filtered_df_time_test.xlsx')
filtered_df_family_test.to_excel(r'C:\Users\1\Desktop\archive\slovo\filtered_df_family_test.xlsx')

output_folder = r'C:\Users\1\Desktop\archive\slovo\all_time_test'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

video_files = os.listdir(r'C:\Users\1\Desktop\test')
for video_file in video_files:
    video_name = os.path.splitext(video_file)[0]
    if video_name in filtered_df_time_test['attachment_id'].values:
        video_path = os.path.join(r'C:\Users\1\Desktop\test', video_file)
        output_path = os.path.join(output_folder, video_file)
        os.rename(video_path, output_path)

output_folder = r'C:\Users\1\Desktop\archive\slovo\all_fam_test'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

video_files = os.listdir(r'C:\Users\1\Desktop\test')
for video_file in video_files:
    video_name = os.path.splitext(video_file)[0]
    if video_name in filtered_df_family_test['attachment_id'].values:
        video_path = os.path.join(r'C:\Users\1\Desktop\test', video_file)
        output_path = os.path.join(output_folder, video_file)
        os.rename(video_path, output_path)

print("done")

