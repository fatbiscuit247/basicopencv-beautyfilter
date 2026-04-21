import cv2
import importlib
import filter as filter_module

# Button dimensions
BTN_X, BTN_Y, BTN_W, BTN_H = 10, 10, 180, 40

filter_on = True

def mouse_callback(event, x, y, flags, param):
    global filter_on
    if event == cv2.EVENT_LBUTTONDOWN:
        if BTN_X <= x <= BTN_X + BTN_W and BTN_Y <= y <= BTN_Y + BTN_H:
            filter_on = not filter_on

def draw_button(frame, is_on):
    color = (80, 180, 80) if is_on else (80, 80, 180)
    label = "Filter: ON" if is_on else "Filter: OFF"
    cv2.rectangle(frame, (BTN_X, BTN_Y), (BTN_X + BTN_W, BTN_Y + BTN_H), color, -1)
    cv2.rectangle(frame, (BTN_X, BTN_Y), (BTN_X + BTN_W, BTN_Y + BTN_H), (255, 255, 255), 2)
    cv2.putText(frame, label, (BTN_X + 12, BTN_Y + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Beauty filter running.")
    print("  q = quit | s = save snapshot | r = reload filter | click button to toggle")

    cv2.namedWindow("opencv-beautyfilter")
    cv2.setMouseCallback("opencv-beautyfilter", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = filter_module.apply_beauty_filter(frame) if filter_on else frame.copy()
        draw_button(output, filter_on)
        cv2.imshow("opencv-beautyfilter", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("snapshot.jpg", output)
            print("Snapshot saved.")
        elif key == ord('r'):
            importlib.reload(filter_module)
            print("Filter reloaded.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()