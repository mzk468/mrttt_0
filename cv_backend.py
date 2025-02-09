import cv2
import numpy as np
import mediapipe as mp
import json
import time

# Medipipe can go fuck itself it took me 7 years to figure out the DLL install
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Load custom images for X and O
x_image = cv2.imread("mushroom.png", cv2.IMREAD_UNCHANGED)
o_image = cv2.imread("berries.png", cv2.IMREAD_UNCHANGED)

# Convert images to include alpha channel if not already present
if x_image.shape[2] == 3:
    x_image = cv2.cvtColor(x_image, cv2.COLOR_BGR2BGRA)
if o_image.shape[2] == 3:
    o_image = cv2.cvtColor(o_image, cv2.COLOR_BGR2BGRA)

# Current board state
grid_state = [[None, None, None], [None, None, None], [None, None, None]]
current_player = "x"

# Function to convert game state to JSON
def game_state_to_json():
    return json.dumps({"current_player": current_player, "game_state": grid_state})

# Resize image to fit each cell on the board
def resize_image_to_cell(image, cell_width, cell_height):
    return cv2.resize(image, (cell_width, cell_height))

# Overlay an image onto a frame
def overlay_image(background, overlay, x, y, width, height):
    overlay_resized = resize_image_to_cell(overlay, width, height)

    if overlay_resized.shape[2] == 4:
        alpha_channel = overlay_resized[:, :, 3] / 255.0  # Normalize to [0, 1]
        overlay_resized = overlay_resized[:, :, :3]  # Drop alpha channel for blending
    else:
        alpha_channel = np.ones((overlay_resized.shape[0], overlay_resized.shape[1]))

    # Get the region of interest (ROI) on the background frame
    roi = background[y:y + height, x:x + width]
    
    # Blend the images based on the alpha channel
    for c in range(0, 3):
        roi[:, :, c] = alpha_channel * overlay_resized[:, :, c] + (1 - alpha_channel) * roi[:, :, c]
    
    background[y:y + height, x:x + width] = roi
    return background

# Draw the Tic-Tac-Toe board
def draw_tictactoe_board(frame):
    board_size = (frame.shape[1], frame.shape[0])  # WxH
    grid_size = 3
    cell_width = board_size[0] // grid_size
    cell_height = board_size[1] // grid_size

    # Draw board grid
    for i in range(1, grid_size):
        # Vertical lines
        cv2.line(frame, (i * cell_width, 0), (i * cell_width, board_size[1]), (255, 0, 0), 5)
        # Horizontal lines
        cv2.line(frame, (0, i * cell_height), (board_size[0], i * cell_height), (255, 0, 0), 5)

    # Place X and O images
    for i in range(3):
        for j in range(3):
            if grid_state[i][j] == "x":
                resized_x_image = resize_image_to_cell(x_image, cell_width, cell_height)
                frame = overlay_image(frame, resized_x_image, j * cell_width, i * cell_height, cell_width, cell_height)
            elif grid_state[i][j] == "o":
                resized_o_image = resize_image_to_cell(o_image, cell_width, cell_height)
                frame = overlay_image(frame, resized_o_image, j * cell_width, i * cell_height, cell_width, cell_height)

    return frame

# Get the grid cell from the hand position
def get_cell_from_hand_position(hand_landmarks, frame):
    board_size = (frame.shape[1], frame.shape[0])  # WxH
    grid_size = 3
    cell_width = board_size[0] // grid_size
    cell_height = board_size[1] // grid_size

    # Get index finger tip position (Landmark 8)
    index_finger_tip = hand_landmarks[8]
    x, y = index_finger_tip.x * frame.shape[1], index_finger_tip.y * frame.shape[0]

    # Map the finger position to grid cell
    row = int(y // cell_height)
    col = int(x // cell_width)

    return row, col

# Check if hand is pointing to a valid grid cell
def is_hand_pointing_to_grid(hand_landmarks, frame):
    row, col = get_cell_from_hand_position(hand_landmarks, frame)
    return 0 <= row < 3 and 0 <= col < 3

# Update the game state with the player's move
def update_game_state(row, col):
    global current_player

    if grid_state[row][col] is None:
        grid_state[row][col] = current_player
        current_player = "o" if current_player == "x" else "x"

# Open video stream with AR board
def open_videostream_with_ar_board():
    global current_player
    cap = cv2.VideoCapture(2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a more natural feel
        frame = cv2.flip(frame, 1)

        # Convert the BGR frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check if hand is pointing to a valid grid cell
                if is_hand_pointing_to_grid(hand_landmarks.landmark, frame):
                    row, col = get_cell_from_hand_position(hand_landmarks.landmark, frame)

                    # Update game state with the player's move
                    update_game_state(row, col)
                    print(f"Move registered at cell ({row}, {col})")

        # Draw the Tic-Tac-Toe board with the updated state
        frame_with_board = draw_tictactoe_board(frame)

        # Display the frame with the AR board
        cv2.imshow("Tic Tac Toe", frame_with_board)
        print(game_state_to_json())

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    open_videostream_with_ar_board()

