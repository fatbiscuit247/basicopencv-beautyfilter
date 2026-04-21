import cv2
from filter import apply_beauty_filter

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Beauty filter running. Press 'q' to quit, 's' to save a snapshot.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        output = apply_beauty_filter(frame)

        cv2.imshow("Original", frame)
        cv2.imshow("Beauty Filter", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("snapshot.jpg", output)
            print("Snapshot saved as snapshot.jpg")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
