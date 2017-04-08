from moviepy.editor import VideoFileClip

from source.processor import process_image

project_video = '../project_video.mp4'
project_output = '../output_video.mp4'

challenge_video = '../challenge_video.mp4'
challenge_output = '../challenge_output.mp4'

harder_challenge_video = '../harder_challenge_video.mp4'
harder_challenge_output = '../harder_challenge_output.mp4'


def process_video(original, processed):
    clip = VideoFileClip(original)
    video_clip = clip.fl_image(process_image)
    video_clip.write_videofile(processed, audio=False)


process_video(project_video, project_output)
