import cv2
import numpy as np
import mediapipe as mp
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import sys
import os

# Suppress TensorFlow and MediaPipe logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # ✅ Drawing utility

hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open webcam
cap = cv2.VideoCapture(0)

# Global rotation and zoom variables
rotate_x = 0
rotate_y = 0
zoom = 0


def display():
    global rotate_x, rotate_y, zoom

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # Camera setup
    gluLookAt(0, 0, 5 + zoom, 0, 0, 0, 0, 1, 0)

    # Apply rotation
    glRotatef(rotate_x, 1, 0, 0)
    glRotatef(rotate_y, 0, 1, 0)

    draw_cube()

    glutSwapBuffers()


def draw_cube():
    glBegin(GL_QUADS)

    # Front face
    glColor3f(1, 0, 0)
    glVertex3f(-1, -1, 1)
    glVertex3f(1, -1, 1)
    glVertex3f(1, 1, 1)
    glVertex3f(-1, 1, 1)

    # Back face
    glColor3f(0, 1, 0)
    glVertex3f(-1, -1, -1)
    glVertex3f(-1, 1, -1)
    glVertex3f(1, 1, -1)
    glVertex3f(1, -1, -1)

    # Left face
    glColor3f(0, 0, 1)
    glVertex3f(-1, -1, -1)
    glVertex3f(-1, -1, 1)
    glVertex3f(-1, 1, 1)
    glVertex3f(-1, 1, -1)

    # Right face
    glColor3f(1, 1, 0)
    glVertex3f(1, -1, -1)
    glVertex3f(1, 1, -1)
    glVertex3f(1, 1, 1)
    glVertex3f(1, -1, 1)

    # Top face
    glColor3f(0, 1, 1)
    glVertex3f(-1, 1, -1)
    glVertex3f(-1, 1, 1)
    glVertex3f(1, 1, 1)
    glVertex3f(1, 1, -1)

    # Bottom face
    glColor3f(1, 0, 1)
    glVertex3f(-1, -1, -1)
    glVertex3f(1, -1, -1)
    glVertex3f(1, -1, 1)
    glVertex3f(-1, -1, 1)

    glEnd()


def idle():
    global rotate_x, rotate_y, zoom

    success, frame = cap.read()
    if success:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # ✅ Draw landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

        # Show webcam with landmarks
        cv2.imshow("Webcam Feed with Landmarks", frame)

        # Control only if two hands detected
        if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:
            hand1 = result.multi_hand_landmarks[0].landmark
            hand2 = result.multi_hand_landmarks[1].landmark

            x1, y1 = hand1[8].x, hand1[8].y
            x2, y2 = hand2[8].x, hand2[8].y

            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

            rotate_x = (0.5 - y_center) * 180
            rotate_y = (x_center - 0.5) * 180
            zoom = np.interp(distance, [0.05, 0.4], [3, -3])  # Adjust if needed

    if cv2.waitKey(1) & 0xFF == 27:
        cap.release()
        cv2.destroyAllWindows()
        sys.exit()

    glutPostRedisplay()


def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(1000, 1000)
    glutCreateWindow(b"3D Cube with Hand Control and Landmarks")
    glEnable(GL_DEPTH_TEST)

    glClearColor(0.1, 0.1, 0.1, 1.0)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1.0, 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)

    glutDisplayFunc(display)
    glutIdleFunc(idle)
    glutMainLoop()


if __name__ == "__main__":
    main()