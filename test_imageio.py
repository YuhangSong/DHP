import pylab
import imageio
import cv2
import numpy as np
filename = '/home/s/ff/vr_new/CS.mp4'
video = imageio.get_reader(filename,  'ffmpeg')
nums = [0, 1]

# for num in nums:
#     image = video.get_data(num)
#     fig = pylab.figure()
#     fig.suptitle('image #{}'.format(num), fontsize=20)
#     pylab.imshow(image)
#
# frame_per_second = video._meta['fps']
# frame_total = video._meta['nframes']
# size = (video._meta['source_size'])
# video_size_width = int(size[0])
# video_size_heigth = int(size[1])
#
# # self.video_size_heigth = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
#
# print(frame_total,frame_per_second, size,video_size_width, video_size_heigth )
#
# pylab.show()

heatmap = np.ones((180,360))
path = '/home/s/ff/'

imageio.imwrite(path+'_'+'test_cv2' + '.jpg',heatmap)
