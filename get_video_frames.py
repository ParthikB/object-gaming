import cv2

VID_PATH = ''
FRAME_SKIP = 5



vid = cv2.VideoCapture(VID_PATH)

frame_number = 0
frames_saved = 0
while True:
	ret, frame = vid.read()

	if not ret:
		break

	frame_number += 1

	if frame_number % FRAME_SKIP == 0:
		frames_saved += 1
		file_name = str(np.random.rand()) + '.jpg'

		cv2.imwrite(file_name, frame)

print(f'Frames Captured : {frames_saved} || FPS : {FRAME_SKIP}')
